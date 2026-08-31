package coreml

import (
	"fmt"
	"math"
)

// Turning what a model wrote into numbers a Go program can use. It is kept
// apart from the binding because it is the only part of this package that can
// be checked without a Mac -- and it is the part most likely to be quietly
// wrong, since a half float read as bytes looks plausible and is nonsense.

// The pixel formats a one-channel output arrives in. CoreVideo names them with
// four character codes, so they are written here the way they are written
// there.
const (
	fmtGray8   = 'L'<<24 | '0'<<16 | '0'<<8 | '8'
	fmtGray16H = 'L'<<24 | '0'<<16 | '0'<<8 | 'h'
	fmtGray32F = 'L'<<24 | '0'<<16 | '0'<<8 | 'f'
)

// decodePlane reads one channel out of a locked pixel buffer.
//
// bpr is the buffer's own stride, which is NOT width times the pixel size:
// CoreVideo pads rows for alignment, and a decoder that assumes otherwise
// produces an image that shears a little more on every row.
func decodePlane(mem []byte, bpr, w, h int, format uint32) (Plane, error) {
	var size int
	switch format {
	case fmtGray8:
		size = 1
	case fmtGray16H:
		size = 2
	case fmtGray32F:
		size = 4
	default:
		return Plane{}, fmt.Errorf("coreml: this output is %s, which is not one channel of numbers", fourCC(format))
	}
	if w < 1 || h < 1 {
		return Plane{}, fmt.Errorf("coreml: an output of %dx%d", w, h)
	}
	if need := (h-1)*bpr + w*size; len(mem) < need {
		return Plane{}, fmt.Errorf("coreml: %d bytes for an image needing %d", len(mem), need)
	}
	out := Plane{Width: w, Height: h, Values: make([]float32, w*h)}
	for y := 0; y < h; y++ {
		row := mem[y*bpr:]
		for x := 0; x < w; x++ {
			switch size {
			case 1:
				out.Values[y*w+x] = float32(row[x])
			case 2:
				out.Values[y*w+x] = half(uint16(row[x*2]) | uint16(row[x*2+1])<<8)
			default:
				out.Values[y*w+x] = math.Float32frombits(
					uint32(row[x*4]) | uint32(row[x*4+1])<<8 |
						uint32(row[x*4+2])<<16 | uint32(row[x*4+3])<<24)
			}
		}
	}
	return out, nil
}

// half expands an IEEE binary16 to a float32.
//
// Go has no float16 and the Neural Engine speaks little else, so this is on
// the path of every frame. The subnormal case is not decoration: a depth model
// puts its nearest surfaces at the top of its range and its noise floor at the
// bottom, and reading the bottom as zero flattens exactly the far detail the
// map is for.
func half(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff
	switch {
	case exp == 0 && mant == 0:
		return math.Float32frombits(sign) // zero, either sign
	case exp == 0:
		for mant&0x400 == 0 { // normalise by hand
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3ff
	case exp == 0x1f:
		return math.Float32frombits(sign | 0x7f800000 | mant<<13) // infinity, or not a number
	}
	return math.Float32frombits(sign | (exp+112)<<23 | mant<<13)
}

// Normalised stretches the plane onto 0..255 for looking at.
//
// A depth model's scale is relative — it says which of two things is nearer,
// not how far either is — so an unstretched map of a scene with no sky in it
// is a uniform grey that tells you nothing about whether the model worked.
//
// Values that are not numbers are left at zero rather than poisoning the range.
func (p Plane) Normalised() []byte {
	out := make([]byte, len(p.Values))
	lo, hi := math.Inf(1), math.Inf(-1)
	for _, v := range p.Values {
		f := float64(v)
		if math.IsNaN(f) || math.IsInf(f, 0) {
			continue
		}
		lo, hi = math.Min(lo, f), math.Max(hi, f)
	}
	span := hi - lo
	if math.IsInf(lo, 0) || span <= 0 {
		return out // nothing to stretch: every value the same, or none usable
	}
	for i, v := range p.Values {
		f := float64(v)
		if math.IsNaN(f) || math.IsInf(f, 0) {
			continue
		}
		out[i] = byte(255 * (f - lo) / span)
	}
	return out
}

func fourCC(v uint32) string {
	b := [4]byte{byte(v >> 24), byte(v >> 16), byte(v >> 8), byte(v)}
	for _, c := range b {
		if c < ' ' || c > '~' {
			return fmt.Sprintf("format %d", v)
		}
	}
	return string(b[:])
}
