package myfunc

import (
	"reflect"
	"testing"

	"github.com/gonum/matrix/mat64"
)

func TestSigmoid(t *testing.T) {
	expected := 0.5
	actual := Sigmoid(0)

	if actual != expected {
		t.Errorf("got %v\nwant %v", actual, expected)
	}

}

func TestVec2Slice(t *testing.T) {
	expected := []float64{1, 2, 3, 4}
	v := mat64.NewVector(len(expected), expected)

	actual := Vec2Slice(v)

	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("got %v\nwant %v", actual, expected)
	}
}
