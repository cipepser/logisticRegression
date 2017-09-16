package main

import (
	"fmt"
	"image/color"
	"math/rand"
	"os/exec"
	"time"

	"./myfunc"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/plot"
	"github.com/gonum/plot/plotter"
	"github.com/gonum/plot/vg"
)

var (
	N   = 1000 // 教師データの数
	M   = 3    //特徴量の次元
	eta = 0.1  // 学習率
)

// 真の分離平面 5x+3y=1
func h(x1, x2 float64) float64 {
	return 2*x1 - 3*x2 - 1
}

// 特徴量
func phi(x1, x2 float64) *mat64.Vector {
	return mat64.NewVector(M, []float64{x1, x2, 1})
}

func main() {
	// テストデータの用意
	rand.Seed(time.Now().UnixNano())

	x1 := make([]float64, N)
	x2 := make([]float64, N)
	for i := 0; i < N; i++ {
		x1[i] = 20*rand.Float64() - 10
		x2[i] = 20*rand.Float64() - 10
	}

	// 正解ラベルの作成
	t := make([]float64, N)
	for i := range t {
		if h(x1[i], x2[i]) > 0 {
			t[i] = 1
		}
	}
	// fmt.Println(t)

	// パラメータの初期化
	ws := make([]float64, M)
	for i := range ws {
		ws[i] = rand.Float64()
	}
	w := mat64.NewVector(len(ws), ws)

	// 学習
	for i := 0; i < N; i++ {
		x := phi(x1[i], x2[i])

		p := myfunc.Sigmoid(mat64.Dot(w, x))
		// fmt.Println(p)

		// fmt.Println(eta * (p - t[i]))
		// fmt.Println(x)
		x.ScaleVec(eta*(p-t[i]), x)
		// fmt.Println(x)
		// fmt.Println("------------")
		w.SubVec(w, x)

		// eta *= float64(N) / (1 + float64(N))
		// eta *= 0.999999999999999
	}

	// plot
	p, err := plot.New()
	if err != nil {
		panic(err)
	}

	idx0 := make([]int, 0)
	// cnt0 := 0
	idx1 := make([]int, 0)
	// cnt1 := 0
	for i := 0; i < N; i++ {
		if t[i] == 0 {
			// idx0[cnt0] = i
			// cnt0++
			idx0 = append(idx0, i)
		} else {
			// idx1[cnt1] = i
			// cnt1++
			idx1 = append(idx1, i)
		}
	}

	data0 := make(plotter.XYs, len(idx0))
	for i := 0; i < len(idx0); i++ {
		data0[i].X = x1[idx0[i]]
		data0[i].Y = x2[idx0[i]]
	}

	s, err := plotter.NewScatter(data0)
	if err != nil {
		panic(err)
	}

	s.Radius = vg.Length(2)
	s.Color = color.RGBA{R: 255, B: 128, A: 255}

	p.Add(s)

	data1 := make(plotter.XYs, len(idx1))
	for i := 0; i < len(idx1); i++ {
		data1[i].X = x1[idx1[i]]
		data1[i].Y = x2[idx1[i]]
	}

	s, err = plotter.NewScatter(data1)
	if err != nil {
		panic(err)
	}

	s.Radius = vg.Length(2)
	s.Color = color.RGBA{R: 255, G: 255}

	p.Add(s)

	line := plotter.NewFunction(func(x float64) float64 {
		return -w.At(2, 0)/w.At(1, 0) - w.At(0, 0)*x/w.At(1, 0)
		// return x + 1
	})
	line.Color = color.RGBA{G: 255, A: 255}
	line.Width = vg.Points(2)

	p.Add(line)

	file := "img.png"
	if err = p.Save(10*vg.Inch, 6*vg.Inch, file); err != nil {
		panic(err)
	}

	if err = exec.Command("open", file).Run(); err != nil {
		panic(err)
	}

	// 判別
	correct := 0
	for i := 0; i < N; i++ {
		// x1[i] = 4*rand.Float64() - 2
		// x2[i] = 4*rand.Float64() - 2

		x := phi(x1[i], x2[i])

		label := 0
		// fmt.Println(myfunc.Sigmoid(mat64.Dot(w, x)))
		if myfunc.Sigmoid(mat64.Dot(w, x)) > 0.5 {
			label = 1
		}
		// fmt.Println(label, t[i])

		if t[i] == float64(label) {
			correct++
		}
	}
	fmt.Println(w)

	fmt.Println("training data: ", float64(correct)/float64(N))

	correct = 0
	for i := 0; i < N; i++ {
		x1[i] = 20*rand.Float64() - 10
		x2[i] = 20*rand.Float64() - 10
		if h(x1[i], x2[i]) > 0 {
			t[i] = 1
		} else {
			t[i] = 0
		}

		x := phi(x1[i], x2[i])

		label := 0
		// fmt.Println(myfunc.Sigmoid(mat64.Dot(w, x)))
		if myfunc.Sigmoid(mat64.Dot(w, x)) > 0.5 {
			label = 1
		}
		// fmt.Println(label, t[i])

		if t[i] == float64(label) {
			correct++
		}
	}
	// fmt.Println(w)

	fmt.Println("new data: ", float64(correct)/float64(N))

	myfunc.TryScaleVec()

}
