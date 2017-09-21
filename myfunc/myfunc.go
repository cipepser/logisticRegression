package myfunc

import (
	"math"

	"gonum.org/v1/gonum/mat"
	// "github.com/cipepser/plot/myutil"
)

func Sigmoid(v float64) float64 { return 1.0 / (1.0 + math.Exp(-v)) }

func Vec2Slice(v *mat.VecDense) []float64 {
	s := make([]float64, v.Len())
	for i := 0; i < v.Len(); i++ {
		s[i] = v.At(i, 0)
	}

	return s
}
