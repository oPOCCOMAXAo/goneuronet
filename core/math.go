package core

import (
	"fmt"
)

func Sqr(x float64) float64 {
	return x * x
}

func FloatToFixed(f float64) string {
	return fmt.Sprintf("%6.3f", f)
}
