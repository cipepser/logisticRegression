package main

import (
	"fmt"
	"image/color"
	"math"
	"math/rand"
	"os/exec"
	"time"

	"./myfunc"
	"github.com/gonum/plot"
	"github.com/gonum/plot/plotter"
	"github.com/gonum/plot/vg"
	"gonum.org/v1/gonum/mat"
)

var (
	N   = 1000 // 教師データの数
	M   = 3    //特徴量の次元
	eta = 0.1  // 学習率
)

func Sigmoid(v float64) float64 { return 1.0 / (1.0 + math.Exp(-v)) }

// 真の分離平面 2x-3y=1
func h(x1, x2 float64) float64 {
	return 2*x1 - 3*x2 - 1
}

// 特徴量
func phi(x1, x2 float64) *mat.VecDense {
	return mat.NewVecDense(M, []float64{1, x1, x2})
}

func main() {
	// テストデータの用意
	rand.Seed(time.Now().UnixNano())

	x1 := make([]float64, N)
	x2 := make([]float64, N)
	for i := 0; i < N; i++ {
		x1[i] = 10*rand.Float64() - 5
		x2[i] = 10*rand.Float64() - 5
	}

	// 正解ラベルの作成
	t := make([]float64, N)
	for i := range t {
		if h(x1[i], x2[i]) > 0 {
			t[i] = 1
		}
	}

	// パラメータの初期化
	ws := make([]float64, M)
	for i := range ws {
		ws[i] = rand.Float64()
	}
	w := mat.NewVecDense(len(ws), ws)

	// 学習
	for i := 0; i < N; i++ {
		x := phi(x1[i], x2[i])
		p := myfunc.Sigmoid(mat.Dot(w, x))

		x.ScaleVec(eta*(p-t[i]), x)
		w.SubVec(w, x)

		eta *= 0.999
	}

	// plot
	p, err := plot.New()
	if err != nil {
		panic(err)
	}

	idx0 := make([]int, 0)
	idx1 := make([]int, 0)
	for i := 0; i < N; i++ {
		if t[i] == 0 {
			idx0 = append(idx0, i)
		} else {
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
		return -w.At(1, 0)*x/w.At(2, 0) - w.At(0, 0)/w.At(2, 0)
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
		x := phi(x1[i], x2[i])

		label := 0
		if myfunc.Sigmoid(mat.Dot(w, x)) > 0.5 {
			label = 1
		}

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
		if myfunc.Sigmoid(mat.Dot(w, x)) > 0.5 {
			label = 1
		}

		if t[i] == float64(label) {
			correct++
		}
	}

	fmt.Println("new data: ", float64(correct)/float64(N))
}
