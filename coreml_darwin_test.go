package coreml

import (
	"math"
	"math/rand/v2"
	"os"
	"testing"
)

func TestTheBindingRefusesWhatItCannotDo(t *testing.T) {
	// None of this needs a model, which is the point: these are the paths a
	// caller hits on a bad day, and they are the ones least likely to be
	// exercised by the happy case.
	if err := Compile(t.TempDir()+"/absent.mlpackage", t.TempDir()+"/out.mlmodelc"); err == nil {
		t.Error("a model that is not there compiled")
	}
	if _, err := Open(t.TempDir()+"/absent.mlmodelc", All); err == nil {
		t.Error("a model that is not there opened")
	}
	// A directory that exists but is not a model: the file check passes and
	// Core ML itself has to refuse it.
	if _, err := Open(t.TempDir(), All); err == nil {
		t.Error("an empty directory opened as a model")
	}

	var m *Model
	if _, err := m.Predict(map[string]Value{"image": Image(make([]byte, 4), 1, 1)}); err == nil {
		t.Error("a nil model predicted")
	}
	m.Close()

	var r *Result
	if _, err := r.Plane("depth"); err == nil {
		t.Error("a closed result gave a plane")
	}
	r.Close()
}

// Compile already having been done is not an error: a program that calls it on
// every start must pay the seconds only once.
func TestCompilingSomethingAlreadyThereIsNotAnError(t *testing.T) {
	dst := t.TempDir() + "/already.mlmodelc"
	if err := os.WriteFile(dst, []byte("pretend"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := Compile("/does/not/matter", dst); err != nil {
		t.Fatalf("Compile over an existing result: %v", err)
	}
}

func TestAnInputThatIsNotAPictureIsRefused(t *testing.T) {
	// pixelBuffer is reached through Predict, but its complaints are about the
	// caller's slice rather than about Core ML, so they are worth their own
	// test: a picture whose bytes do not match its stated size is the most
	// likely mistake, and reading past it would be a crash in someone's frame
	// loop rather than an error in their log.
	for _, tc := range []struct {
		name string
		v    Value
	}{
		{"no width", Image(make([]byte, 16), 0, 2)},
		{"no height", Image(make([]byte, 16), 2, 0)},
		{"fewer bytes than pixels", Image(make([]byte, 8), 4, 4)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := load(); err != nil {
				t.Skip(err)
			}
			if _, err := pixelBuffer(tc.v); err == nil {
				t.Fatal("it was accepted")
			}
		})
	}
	if err := load(); err == nil {
		if pb, err := pixelBuffer(Image(make([]byte, 64), 4, 4)); err != nil {
			t.Errorf("a valid picture was refused: %v", err)
		} else {
			cvRel(pb)
		}
	}
}

// The live test. It needs a compiled model, which a build machine has no
// reason to carry, so it is asked for by name and skipped when absent:
//
//	COREML_TEST_MODEL=/path/to/Something.mlmodelc go test ./...
func TestARealModelRunsAndAnswersWithNumbersThatVary(t *testing.T) {
	path := os.Getenv("COREML_TEST_MODEL")
	if path == "" {
		t.Skip("set COREML_TEST_MODEL to a compiled .mlmodelc to run this")
	}
	for _, units := range []Units{CPUOnly, CPUAndGPU, CPUAndNeuralEngine, All} {
		m, err := Open(path, units)
		if err != nil {
			t.Fatalf("units %d: %v", units, err)
		}
		defer m.Close()

		in := m.Inputs()
		if len(in) == 0 || in[0].Kind != KindImage {
			t.Fatalf("this model takes %+v, and the test needs an image", in)
		}
		w, h := in[0].Width, in[0].Height
		if w < 1 || h < 1 {
			t.Fatalf("the model asked for a %dx%d image", w, h)
		}
		pix := make([]byte, w*h*4)
		r := rand.New(rand.NewPCG(1, 2))
		for i := 0; i < len(pix); i += 4 {
			v := byte(r.Uint32())
			pix[i], pix[i+1], pix[i+2], pix[i+3] = v, v, v, 255
		}

		res, err := m.Predict(map[string]Value{in[0].Name: Image(pix, w, h)})
		if err != nil {
			t.Fatalf("units %d: %v", units, err)
		}
		out := m.Outputs()
		if len(out) == 0 {
			t.Fatal("this model returns nothing")
		}
		p, err := res.Plane(out[0].Name)
		if err != nil {
			t.Fatalf("units %d: %v", units, err)
		}
		res.Close()

		if p.Width < 1 || len(p.Values) != p.Width*p.Height {
			t.Fatalf("a plane of %dx%d with %d values", p.Width, p.Height, len(p.Values))
		}
		// A model that ran and returned one flat number is a model that did
		// not work -- and it keeps perfect time, so only the values say so.
		lo, hi := math.Inf(1), math.Inf(-1)
		for _, v := range p.Values {
			lo, hi = math.Min(lo, float64(v)), math.Max(hi, float64(v))
		}
		if !(hi > lo) {
			t.Fatalf("units %d: every value is %v", units, lo)
		}
		if _, err := res.Plane(out[0].Name); err == nil {
			t.Error("a closed result still gave a plane")
		}
		if _, err := res.Plane("nothing-is-called-this"); err == nil {
			t.Error("a name the model does not have gave a plane")
		}
	}
}
