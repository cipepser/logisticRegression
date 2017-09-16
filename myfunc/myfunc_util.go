package myfunc

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/cipepser/plot/myutil"
	"github.com/gonum/matrix/mat64"
)

func Scatter() {
	rand.Seed(time.Now().UnixNano())

	x := make([]float64, 100)
	y := make([]float64, len(x))
	for i := range x {
		x[i] = float64(i) * 0.1
		y[i] = x[i] + rand.Float64()
	}

	myutil.MyScatter(x, y)
}

func TryMat() {
	a := mat64.NewDense(2, 2, []float64{1, 2, 3, 4})
	fmt.Println(a)

	b := mat64.NewDense(2, 2, []float64{1, 2, 3, 4})
	a.Add(a, b)
	fmt.Println(a)
}

func TryVec() {
	a := mat64.NewDense(2, 2, []float64{1, 2, 3, 4})

	b := a.ColView(1)
	fmt.Println(b.At(0, 0))
	fmt.Println(b.At(1, 0))
	fmt.Println(a.RowView(1))
}

func TryDotProduct() {
	v := mat64.NewVector(4, []float64{1, 2, 3, 4})
	w := mat64.NewVector(4, []float64{1, 2, 3, 4})
	p := mat64.Dot(v, w)
	fmt.Println(p)
}

func TryVecSub() {
	v := mat64.NewVector(4, []float64{1, 2, 3, 4})
	w := mat64.NewVector(4, []float64{1, 1, 1, 1})

	v.SubVec(v, w)
	fmt.Println(v)

}

func TryScaleVec() {
	v := mat64.NewVector(4, []float64{1, 2, 3, 4})
	v.ScaleVec(2, v)

	fmt.Println(v)
}
