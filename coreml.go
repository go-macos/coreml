// Package coreml runs a trained model on a Mac's Neural Engine from pure Go,
// with no cgo and no Xcode.
//
// The Neural Engine is the part of an Apple chip that costs almost nothing to
// use. Depth Anything V2 Small, on an M4 Max, one 518x392 frame:
//
//	processor only              46 ms per frame, 114 ms of processor time
//	processor and GPU           13 ms per frame,   5.1 ms of processor time
//	processor and Neural Engine 23 ms per frame,   0.4 ms of processor time
//
// The Neural Engine is not the fastest of the three. It is the one that leaves
// the machine alone: a frame costs four tenths of a millisecond of CPU, which
// is nearly three hundred times less than doing it on the processor. On a
// laptop that is also drawing a browser and syncing files, that is the number
// a person feels.
//
//	// once, and it is kept: compiling is slow and the result is durable
//	err := coreml.Compile("Depth.mlpackage", "Depth.mlmodelc")
//
//	m, err := coreml.Open("Depth.mlmodelc", coreml.CPUAndNeuralEngine)
//	defer m.Close()
//
//	in := m.Inputs()[0]              // says what size the model wants
//	res, err := m.Predict(map[string]Value{in.Name: coreml.Image(bgra, in.Width, in.Height)})
//	defer res.Close()
//
//	plane, err := res.Plane("depth") // one channel, as float32
//
// # A model has to be compiled before it can be opened
//
// What a model is distributed as — an .mlpackage — is not what CoreML runs.
// Compile turns one into the other using the system compiler, at run time,
// with nothing installed. It takes seconds, so the result belongs somewhere
// durable: macOS compiles into a temporary directory it is free to empty, and
// a program that does not move the result out pays that cost on every start.
//
// # Everywhere else
//
// On a platform that is not macOS every constructor returns ErrUnsupported, so
// a program that offers a Neural Engine path and a portable one cross-compiles
// without build tags of its own.
package coreml

import "errors"

// ErrUnsupported is returned on any platform that is not macOS.
var ErrUnsupported = errors.New("coreml: only macOS has Core ML")

// Units says which processors a model may run on.
//
// It is a request, not an instruction: Core ML decides for itself, per layer,
// and a model with a layer the Neural Engine cannot do will run that layer
// elsewhere. Asking for CPUAndNeuralEngine is how a model reaches the Neural
// Engine at all — the default leaves it out.
type Units int

const (
	// CPUOnly is the slowest and the most predictable.
	CPUOnly Units = 0
	// CPUAndGPU is usually the fastest in wall-clock time.
	CPUAndGPU Units = 1
	// All lets Core ML choose, which in practice means mostly the GPU.
	All Units = 2
	// CPUAndNeuralEngine is the cheapest in processor time by a wide margin.
	CPUAndNeuralEngine Units = 3
)

// Kind is what sort of thing a model takes or returns.
type Kind int

const (
	// KindOther is anything this package does not yet describe: a dictionary,
	// a sequence, a number. Naming it is more useful than leaving it out,
	// because a model whose input is KindOther cannot be fed from here and the
	// caller should be told so rather than left guessing.
	KindOther Kind = iota
	// KindImage is a picture, and the only input this package can supply.
	KindImage
	// KindMultiArray is a tensor.
	KindMultiArray
)

// String names a kind, so that an error about the wrong one reads.
func (k Kind) String() string {
	switch k {
	case KindImage:
		return "image"
	case KindMultiArray:
		return "multi-array"
	}
	return "something this package does not handle"
}

// Feature describes one input or output of a model. Width and Height are set
// only for an image, and are what the model demands: a picture of any other
// size is refused, so this is the size to scale to.
type Feature struct {
	Name          string
	Kind          Kind
	Width, Height int
}

// Model is a compiled model, open and ready to predict.
type Model struct {
	id      uintptr
	inputs  []Feature
	outputs []Feature
}

// Inputs is what the model takes.
func (m *Model) Inputs() []Feature {
	if m == nil {
		return nil
	}
	return m.inputs
}

// Outputs is what the model returns.
func (m *Model) Outputs() []Feature {
	if m == nil {
		return nil
	}
	return m.outputs
}

// Value is one input. Build one with Image.
type Value struct {
	pix           []byte
	width, height int
}

// Image is a picture to feed a model: four bytes per pixel, blue green red
// alpha, no padding between rows. That order is CoreVideo's, and it is what a
// screen capture and a decoded video frame already are on this platform.
func Image(bgra []byte, width, height int) Value {
	return Value{pix: bgra, width: width, height: height}
}

// Result is what a prediction returned. Close it.
type Result struct {
	id uintptr
}

// Plane is one channel of an image output, as float32 whatever the model
// stored it as.
type Plane struct {
	Width, Height int
	Values        []float32
}
