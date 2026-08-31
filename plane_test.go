package coreml

import (
	"math"
	"testing"
)

func TestHalfExpandsEveryClassOfBinary16(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   uint16
		want float32
	}{
		{"zero", 0x0000, 0},
		{"minus zero", 0x8000, float32(math.Copysign(0, -1))},
		{"one", 0x3C00, 1},
		{"minus two", 0xC000, -2},
		{"a half", 0x3800, 0.5},
		// The smallest subnormal. Read by the naive formula it comes out as
		// zero, and a depth model's far detail lives down here.
		{"smallest subnormal", 0x0001, 5.9604645e-08},
		{"largest subnormal", 0x03FF, 6.0975552e-05},
		{"smallest normal", 0x0400, 6.1035156e-05},
		{"largest finite", 0x7BFF, 65504},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := half(tc.in)
			if got != tc.want {
				t.Fatalf("half(%#04x) = %v, want %v", tc.in, got, tc.want)
			}
			if math.Signbit(float64(got)) != math.Signbit(float64(tc.want)) {
				t.Fatalf("half(%#04x) has the wrong sign", tc.in)
			}
		})
	}
	if got := half(0x7C00); !math.IsInf(float64(got), 1) {
		t.Errorf("half(0x7C00) = %v, want +Inf", got)
	}
	if got := half(0xFC00); !math.IsInf(float64(got), -1) {
		t.Errorf("half(0xFC00) = %v, want -Inf", got)
	}
	if got := half(0x7E00); !math.IsNaN(float64(got)) {
		t.Errorf("half(0x7E00) = %v, want NaN", got)
	}
}

func TestDecodeReadsEachOneChannelFormat(t *testing.T) {
	// Two pixels wide, two tall, with a stride WIDER than the pixels -- which
	// is what CoreVideo actually hands back. A decoder that assumes the stride
	// is the width reads the padding as picture and shears the image.
	t.Run("eight bit, padded rows", func(t *testing.T) {
		mem := []byte{1, 2, 0xFF, 0xFF, 3, 4, 0xFF, 0xFF}
		p, err := decodePlane(mem, 4, 2, 2, fmtGray8)
		if err != nil {
			t.Fatal(err)
		}
		want := []float32{1, 2, 3, 4}
		for i := range want {
			if p.Values[i] != want[i] {
				t.Fatalf("values = %v, want %v", p.Values, want)
			}
		}
	})
	t.Run("half float, padded rows", func(t *testing.T) {
		mem := []byte{0x00, 0x3C, 0x00, 0x40, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0xC0, 0xFF, 0xFF}
		p, err := decodePlane(mem, 6, 2, 2, fmtGray16H)
		if err != nil {
			t.Fatal(err)
		}
		want := []float32{1, 2, 0, -2}
		for i := range want {
			if p.Values[i] != want[i] {
				t.Fatalf("values = %v, want %v", p.Values, want)
			}
		}
	})
	t.Run("thirty-two bit float", func(t *testing.T) {
		mem := make([]byte, 8)
		for i, v := range []float32{1.5, -0.25} {
			b := math.Float32bits(v)
			mem[i*4], mem[i*4+1], mem[i*4+2], mem[i*4+3] = byte(b), byte(b>>8), byte(b>>16), byte(b>>24)
		}
		p, err := decodePlane(mem, 8, 2, 1, fmtGray32F)
		if err != nil {
			t.Fatal(err)
		}
		if p.Values[0] != 1.5 || p.Values[1] != -0.25 {
			t.Fatalf("values = %v", p.Values)
		}
	})
}

func TestDecodeRefusesWhatItCannotRead(t *testing.T) {
	for _, tc := range []struct {
		name      string
		mem       []byte
		bpr, w, h int
		format    uint32
	}{
		{"a colour output", make([]byte, 64), 8, 2, 2, 'B'<<24 | 'G'<<16 | 'R'<<8 | 'A'},
		{"a format with no name", make([]byte, 64), 8, 2, 2, 7},
		{"no width", make([]byte, 64), 8, 0, 2, fmtGray8},
		{"no height", make([]byte, 64), 8, 2, 0, fmtGray8},
		{"a buffer too short for the image", make([]byte, 3), 4, 2, 2, fmtGray8},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := decodePlane(tc.mem, tc.bpr, tc.w, tc.h, tc.format); err == nil {
				t.Fatal("it was accepted")
			}
		})
	}
	// The negative control: the same call, valid, still works. Without it a
	// decoder that refused everything would pass the test above.
	if _, err := decodePlane(make([]byte, 8), 4, 2, 2, fmtGray8); err != nil {
		t.Fatalf("a valid plane was refused: %v", err)
	}
}

func TestNormalisedStretchesAndSurvivesTheDegenerateCases(t *testing.T) {
	p := Plane{Width: 4, Height: 1, Values: []float32{2, 4, 6, 8}}
	got := p.Normalised()
	if got[0] != 0 || got[3] != 255 {
		t.Fatalf("normalised = %v, want it to reach both ends", got)
	}
	if got[1] != 85 || got[2] != 170 {
		t.Fatalf("normalised = %v, want the middle spaced evenly", got)
	}

	flat := Plane{Width: 2, Height: 1, Values: []float32{3, 3}}
	if g := flat.Normalised(); g[0] != 0 || g[1] != 0 {
		t.Errorf("a flat plane stretched to %v; there is nothing to stretch", g)
	}

	// A value that is not a number must not poison the range: one NaN taken
	// into the minimum turns the whole map black.
	nan := float32(math.NaN())
	mixed := Plane{Width: 3, Height: 1, Values: []float32{nan, 0, 10}}
	g := mixed.Normalised()
	if g[0] != 0 || g[1] != 0 || g[2] != 255 {
		t.Errorf("normalised = %v, want the NaN ignored and the rest stretched", g)
	}

	none := Plane{Width: 2, Height: 1, Values: []float32{nan, float32(math.Inf(1))}}
	if g := none.Normalised(); g[0] != 0 || g[1] != 0 {
		t.Errorf("a plane of no usable values gave %v", g)
	}
}

func TestFourCCNamesWhatItCanAndNumbersWhatItCannot(t *testing.T) {
	if got := fourCC(fmtGray16H); got != "L00h" {
		t.Errorf("fourCC = %q, want L00h", got)
	}
	if got := fourCC(7); got == "" || got[0] != 'f' {
		t.Errorf("fourCC(7) = %q, want it spelled as a number", got)
	}
}

func TestAKindNamesItself(t *testing.T) {
	for k, want := range map[Kind]string{
		KindImage:      "image",
		KindMultiArray: "multi-array",
		KindOther:      "something this package does not handle",
	} {
		if got := k.String(); got != want {
			t.Errorf("Kind(%d) = %q, want %q", k, got, want)
		}
	}
}

func TestTheAccessorsAreSafeOnNothingAtAll(t *testing.T) {
	// A model that failed to open is a nil pointer, and a caller that forgot
	// to check the error will ask it what it takes. That must answer, not
	// crash: the error it already ignored is the thing to fix, and a panic
	// here hides it.
	var m *Model
	if m.Inputs() != nil || m.Outputs() != nil {
		t.Error("a nil model described itself")
	}

	m = &Model{
		inputs:  []Feature{{Name: "image", Kind: KindImage, Width: 518, Height: 392}},
		outputs: []Feature{{Name: "depth", Kind: KindImage, Width: 518, Height: 392}},
	}
	if got := m.Inputs(); len(got) != 1 || got[0].Name != "image" || got[0].Width != 518 {
		t.Errorf("inputs = %+v", got)
	}
	if got := m.Outputs(); len(got) != 1 || got[0].Name != "depth" {
		t.Errorf("outputs = %+v", got)
	}
}

func TestAnImageKeepsWhatItWasGiven(t *testing.T) {
	pix := []byte{1, 2, 3, 4}
	v := Image(pix, 1, 1)
	if v.width != 1 || v.height != 1 || len(v.pix) != 4 {
		t.Fatalf("Image kept %dx%d and %d bytes", v.width, v.height, len(v.pix))
	}
}
