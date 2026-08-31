# coreml

[![Go Reference](https://pkg.go.dev/badge/github.com/go-macos/coreml.svg)](https://pkg.go.dev/github.com/go-macos/coreml)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)
[![Pure Go](https://img.shields.io/badge/pure%20Go-CGO%3D0-00ADD8?logo=go&logoColor=white)](https://github.com/go-macos/coreml)

**Run a trained model on a Mac's Neural Engine, from pure Go, with no cgo and
no Xcode.**

The Neural Engine is the part of an Apple chip that costs almost nothing to
use. Depth Anything V2 Small, on an M4 Max, one 518×392 frame:

| | per frame | processor time per frame |
|---|---|---|
| processor only | 46 ms | 114 ms |
| processor and GPU | **13 ms** | 5.1 ms |
| processor and Neural Engine | 23 ms | **0.4 ms** |

The Neural Engine is not the fastest of the three — the GPU is. It is the one
that leaves the machine alone: a frame costs four tenths of a millisecond of
CPU, nearly three hundred times less than doing it on the processor. On a
laptop that is also drawing a browser and syncing files, that is the number a
person feels.

Which to ask for is therefore a real choice, and this package makes you make
it.

## A model has to be compiled first

What a model is distributed as — an `.mlpackage` — is not what Core ML runs.
`Compile` turns one into the other with the system compiler, at run time, with
nothing installed:

```go
// Seconds, once. Then it is on disk and this call does nothing.
err := coreml.Compile("DepthAnythingV2SmallF16.mlpackage", "Depth.mlmodelc")
```

Where it is kept matters. macOS compiles into a temporary directory it is free
to empty, so a program that does not move the result out pays the seconds again
on every start. `Compile` moves it where you asked.

## Then it is four calls

```go
m, err := coreml.Open("Depth.mlmodelc", coreml.CPUAndNeuralEngine)
defer m.Close()

in := m.Inputs()[0]   // {Name:"image" Kind:image Width:518 Height:392}

res, err := m.Predict(map[string]coreml.Value{
    in.Name: coreml.Image(bgra, in.Width, in.Height),
})
defer res.Close()

plane, err := res.Plane("depth")   // one channel, as float32
png.Encode(w, &image.Gray{Pix: plane.Normalised(), ...})
```

`Inputs` is not decoration. A model refuses a picture of any size but the one
it was trained on, so this is what to scale to — and asking beats hard-coding a
number that changes with the next model.

## Half floats, and why they are the interesting part

The Neural Engine speaks IEEE binary16, and Go has no `float16`. A plane comes
back expanded to `float32`, subnormals included — which is not a detail: a depth
model puts its nearest surfaces at the top of its range and its far detail down
at the bottom, and the naive expansion reads that bottom as zero. Rows are
padded, too, by an amount CoreVideo chooses; a decoder that assumes otherwise
shears the image a little more on every row.

Both of those are portable arithmetic, and both are held at 100% coverage.

## Everywhere else

On any platform that is not macOS, every constructor returns `ErrUnsupported`
and the rest are no-ops. A program that offers a Neural Engine path and a
portable one cross-compiles without build tags of its own.

## What this is not

Images in, images out. Multi-array inputs and outputs, batches, and stateful
models are not here — they are worth adding when something needs them, and not
before.

## Testing against a real model

The live test needs a compiled model, which a build machine has no reason to
carry, so it is asked for by name and skipped when absent:

```
COREML_TEST_MODEL=/path/to/Something.mlmodelc go test ./...
```

Apple publishes CoreML builds of Depth Anything V2 under Apache-2.0, which is
what the numbers above were measured with.

## Install

```
go get github.com/go-macos/coreml
```

CGO_ENABLED=0. The only dependencies are [purego](https://github.com/ebitengine/purego)
and [go-macos/objc](https://github.com/go-macos/objc).
