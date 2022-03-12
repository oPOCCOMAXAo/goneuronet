package core

import (
	"fmt"
	"math"
	"math/rand"
)

type NetDataType float64

func Exp(x NetDataType) NetDataType {
	return NetDataType(math.Exp(float64(x)))
}

func Sqr(x NetDataType) NetDataType {
	return x * x
}

func Abs(x NetDataType) NetDataType {
	if x < 0 {
		return -x
	}

	return x
}

func Rand() NetDataType {
	return NetDataType(rand.Float64())
}

func FloatToFixed(f NetDataType) string {
	return fmt.Sprintf("%6.3f", f)
}
