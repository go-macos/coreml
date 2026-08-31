package coreml

import (
	"fmt"
	"os"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/go-macos/objc"
)

// kCVPixelFormatType_32BGRA, spelled the way CoreVideo spells it.
const fmtBGRA = 'B'<<24 | 'G'<<16 | 'R'<<8 | 'A'

// The Objective-C side is all messages. CoreVideo is not: pixel buffers are C
// functions, which is why they are bound separately.
var (
	loadOnce sync.Once
	loadErr  error

	cvCreate func(alloc uintptr, w, h, format, attrs uintptr, out unsafe.Pointer) int32
	cvLock   func(pb uintptr, flags uint64) int32
	cvUnlock func(pb uintptr, flags uint64) int32
	cvBase   func(pb uintptr) unsafe.Pointer
	cvBPR    func(pb uintptr) uintptr
	cvWidth  func(pb uintptr) uintptr
	cvHeight func(pb uintptr) uintptr
	cvFormat func(pb uintptr) uint32
	cvRel    func(pb uintptr)
)

func load() error {
	loadOnce.Do(func() {
		if err := objc.Load("/System/Library/Frameworks/CoreML.framework/CoreML"); err != nil {
			loadErr = fmt.Errorf("coreml: %w", err)
			return
		}
		h, err := purego.Dlopen("/System/Library/Frameworks/CoreVideo.framework/CoreVideo",
			purego.RTLD_LAZY|purego.RTLD_GLOBAL)
		if err != nil {
			loadErr = fmt.Errorf("coreml: %w", err)
			return
		}
		purego.RegisterLibFunc(&cvCreate, h, "CVPixelBufferCreate")
		purego.RegisterLibFunc(&cvLock, h, "CVPixelBufferLockBaseAddress")
		purego.RegisterLibFunc(&cvUnlock, h, "CVPixelBufferUnlockBaseAddress")
		purego.RegisterLibFunc(&cvBase, h, "CVPixelBufferGetBaseAddress")
		purego.RegisterLibFunc(&cvBPR, h, "CVPixelBufferGetBytesPerRow")
		purego.RegisterLibFunc(&cvWidth, h, "CVPixelBufferGetWidth")
		purego.RegisterLibFunc(&cvHeight, h, "CVPixelBufferGetHeight")
		purego.RegisterLibFunc(&cvFormat, h, "CVPixelBufferGetPixelFormatType")
		purego.RegisterLibFunc(&cvRel, h, "CVPixelBufferRelease")
	})
	return loadErr
}

func nsurl(path string) objc.ID {
	return objc.ClassID("NSURL").Send(objc.Sel("fileURLWithPath:"), objc.NSString(path))
}

func nserr(e objc.ID, fallback string) string {
	if e == 0 {
		return fallback
	}
	if s := objc.Stringify(e.Send(objc.Sel("localizedDescription"))); s != "" {
		return s
	}
	return fallback
}

// Compile turns a distributed .mlpackage into the .mlmodelc that Core ML
// actually runs, and puts it where you asked.
//
// It does nothing if dst already exists, because compiling takes seconds and
// the result does not change. Moving it out of the temporary directory macOS
// compiles into is the whole point: that directory is emptied whenever the
// system feels like it, and a program that leaves the result there pays the
// seconds again on every start.
func Compile(src, dst string) error {
	if err := load(); err != nil {
		return err
	}
	if _, err := os.Stat(dst); err == nil {
		return nil
	}
	if _, err := os.Stat(src); err != nil {
		return fmt.Errorf("coreml: %w", err)
	}
	var e objc.ID
	tmp := objc.ClassID("MLModel").Send(objc.Sel("compileModelAtURL:error:"),
		nsurl(src), unsafe.Pointer(&e))
	if tmp == 0 {
		return fmt.Errorf("coreml: %s", nserr(e, "the model would not compile"))
	}
	e = 0
	fm := objc.ClassID("NSFileManager").Send(objc.Sel("defaultManager"))
	if !objc.Send[bool](fm, objc.Sel("moveItemAtURL:toURL:error:"), tmp, nsurl(dst), unsafe.Pointer(&e)) {
		return fmt.Errorf("coreml: compiled, but could not be kept: %s", nserr(e, "the move failed"))
	}
	return nil
}

// Open loads a compiled model and says which processors it may use.
func Open(path string, units Units) (*Model, error) {
	if err := load(); err != nil {
		return nil, err
	}
	if _, err := os.Stat(path); err != nil {
		return nil, fmt.Errorf("coreml: %w", err)
	}
	cfg := objc.ClassID("MLModelConfiguration").Send(objc.Sel("alloc")).Send(objc.Sel("init"))
	defer cfg.Send(objc.Sel("release"))
	cfg.Send(objc.Sel("setComputeUnits:"), uintptr(units))

	var e objc.ID
	id := objc.ClassID("MLModel").Send(objc.Sel("modelWithContentsOfURL:configuration:error:"),
		nsurl(path), cfg, unsafe.Pointer(&e))
	if id == 0 {
		return nil, fmt.Errorf("coreml: %s", nserr(e, "the model would not open"))
	}
	id.Send(objc.Sel("retain"))
	desc := id.Send(objc.Sel("modelDescription"))
	return &Model{
		id:      uintptr(id),
		inputs:  features(desc.Send(objc.Sel("inputDescriptionsByName"))),
		outputs: features(desc.Send(objc.Sel("outputDescriptionsByName"))),
	}, nil
}

// features reads what a model takes or returns, including the exact size an
// image input demands -- a picture of any other size is refused outright, so
// this is the only thing that says what to scale to.
func features(dict objc.ID) []Feature {
	keys := dict.Send(objc.Sel("allKeys"))
	n := int(keys.Send(objc.Sel("count")))
	out := make([]Feature, 0, n)
	for i := 0; i < n; i++ {
		key := keys.Send(objc.Sel("objectAtIndex:"), uintptr(i))
		d := dict.Send(objc.Sel("objectForKey:"), key)
		f := Feature{Name: objc.Stringify(key)}
		switch objc.Send[int](d, objc.Sel("type")) {
		case 4: // MLFeatureTypeImage
			f.Kind = KindImage
			if c := d.Send(objc.Sel("imageConstraint")); c != 0 {
				f.Width = int(c.Send(objc.Sel("pixelsWide")))
				f.Height = int(c.Send(objc.Sel("pixelsHigh")))
			}
		case 5: // MLFeatureTypeMultiArray
			f.Kind = KindMultiArray
		}
		out = append(out, f)
	}
	return out
}

// Close gives the model back.
func (m *Model) Close() {
	if m == nil || m.id == 0 {
		return
	}
	objc.ID(m.id).Send(objc.Sel("release"))
	m.id = 0
}

// Predict runs the model once.
//
// The picture is copied into memory CoreVideo owns rather than wrapped where
// it lies. It is one copy of a small image against tens of milliseconds of
// inference, and it removes the whole question of what Core ML may still be
// holding when the Go slice is collected.
func (m *Model) Predict(in map[string]Value) (*Result, error) {
	if m == nil || m.id == 0 {
		return nil, fmt.Errorf("coreml: predicting with a closed model")
	}
	if len(in) == 0 {
		return nil, fmt.Errorf("coreml: a prediction with no inputs")
	}
	var res *Result
	var err error
	objc.AutoreleasePool(func() {
		dict := objc.ClassID("NSMutableDictionary").Send(objc.Sel("dictionary"))
		var buffers []uintptr
		defer func() {
			for _, pb := range buffers {
				cvRel(pb)
			}
		}()
		for name, v := range in {
			pb, e := pixelBuffer(v)
			if e != nil {
				err = e
				return
			}
			buffers = append(buffers, pb)
			fv := objc.ClassID("MLFeatureValue").Send(objc.Sel("featureValueWithPixelBuffer:"), pb)
			if fv == 0 {
				err = fmt.Errorf("coreml: %q could not be made into an input", name)
				return
			}
			dict.Send(objc.Sel("setObject:forKey:"), fv, objc.NSString(name))
		}
		var e objc.ID
		prov := objc.ClassID("MLDictionaryFeatureProvider").Send(objc.Sel("alloc")).
			Send(objc.Sel("initWithDictionary:error:"), dict, unsafe.Pointer(&e))
		if prov == 0 {
			err = fmt.Errorf("coreml: %s", nserr(e, "the inputs were refused"))
			return
		}
		defer prov.Send(objc.Sel("release"))
		e = 0
		out := objc.ID(m.id).Send(objc.Sel("predictionFromFeatures:error:"), prov, unsafe.Pointer(&e))
		if out == 0 {
			err = fmt.Errorf("coreml: %s", nserr(e, "the prediction failed"))
			return
		}
		// Retained out of the pool: everything above dies when it drains.
		out.Send(objc.Sel("retain"))
		res = &Result{id: uintptr(out)}
	})
	return res, err
}

func pixelBuffer(v Value) (uintptr, error) {
	if v.width < 1 || v.height < 1 {
		return 0, fmt.Errorf("coreml: an input image of %dx%d", v.width, v.height)
	}
	if need := v.width * v.height * 4; len(v.pix) < need {
		return 0, fmt.Errorf("coreml: %d bytes for a %dx%d image, which needs %d",
			len(v.pix), v.width, v.height, need)
	}
	var pb uintptr
	if r := cvCreate(0, uintptr(v.width), uintptr(v.height), fmtBGRA, 0, unsafe.Pointer(&pb)); r != 0 || pb == 0 {
		return 0, fmt.Errorf("coreml: CoreVideo would not give a %dx%d buffer (%d)", v.width, v.height, r)
	}
	cvLock(pb, 0)
	defer cvUnlock(pb, 0)
	bpr := int(cvBPR(pb))
	dst := unsafe.Slice((*byte)(cvBase(pb)), bpr*v.height)
	for y := 0; y < v.height; y++ {
		copy(dst[y*bpr:y*bpr+v.width*4], v.pix[y*v.width*4:])
	}
	return pb, nil
}

// Plane reads one named output as numbers.
func (r *Result) Plane(name string) (Plane, error) {
	if r == nil || r.id == 0 {
		return Plane{}, fmt.Errorf("coreml: reading from a closed result")
	}
	v := objc.ID(r.id).Send(objc.Sel("featureValueForName:"), objc.NSString(name))
	if v == 0 {
		return Plane{}, fmt.Errorf("coreml: this model returned nothing called %q", name)
	}
	pb := uintptr(v.Send(objc.Sel("imageBufferValue")))
	if pb == 0 {
		return Plane{}, fmt.Errorf("coreml: %q is not an image output", name)
	}
	cvLock(pb, 1) // kCVPixelBufferLock_ReadOnly
	defer cvUnlock(pb, 1)
	bpr, w, h := int(cvBPR(pb)), int(cvWidth(pb)), int(cvHeight(pb))
	mem := unsafe.Slice((*byte)(cvBase(pb)), bpr*h)
	return decodePlane(mem, bpr, w, h, cvFormat(pb))
}

// Close releases the result. The planes already read out of it stay valid:
// they are Go memory.
func (r *Result) Close() {
	if r == nil || r.id == 0 {
		return
	}
	objc.ID(r.id).Send(objc.Sel("release"))
	r.id = 0
}
