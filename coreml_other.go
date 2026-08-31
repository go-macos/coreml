//go:build !darwin

package coreml

// Everywhere that is not macOS. A constructor says so; nothing pretends.

// Compile reports that this platform has no Core ML.
func Compile(string, string) error { return ErrUnsupported }

// Open reports that this platform has no Core ML.
func Open(string, Units) (*Model, error) { return nil, ErrUnsupported }

// Close does nothing.
func (m *Model) Close() {}

// Predict reports that this platform has no Core ML.
func (m *Model) Predict(map[string]Value) (*Result, error) { return nil, ErrUnsupported }

// Plane reports that this platform has no Core ML.
func (r *Result) Plane(string) (Plane, error) { return Plane{}, ErrUnsupported }

// Close does nothing.
func (r *Result) Close() {}
