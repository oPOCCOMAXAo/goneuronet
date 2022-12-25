package core

import (
	"math"
)

type ActivatorClass interface {
	F(float64) float64 // F forward Function
	D(float64) float64 // D backward Derivative
}

type SigmoidActivatorClass struct{}

func (*SigmoidActivatorClass) F(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func (*SigmoidActivatorClass) D(x float64) float64 {
	exp := math.Exp(-x)

	return exp / Sqr(exp+1.0)
}

func CreateSigmoidActivator() ActivatorClass {
	return &SigmoidActivatorClass{}
}

type HardSigmoidActivatorClass struct{}

func (*HardSigmoidActivatorClass) F(value float64) float64 {
	if value < -2.5 {
		return 0
	}

	if value > 2.5 {
		return 1
	}

	return 0.2*value + 0.5
}

func (*HardSigmoidActivatorClass) D(x float64) float64 {
	if x < -2.5 {
		return 0
	}

	if x > 2.5 {
		return 0
	}

	return 0.2
}

func CreateHardSigmoidActivatorClass() ActivatorClass {
	return &HardSigmoidActivatorClass{}
}

type LinearActivatorClass struct{}

func (*LinearActivatorClass) F(x float64) float64 {
	return x
}

func (*LinearActivatorClass) D(x float64) float64 {
	return 1
}

func CreateLinearActivator() ActivatorClass {
	return &LinearActivatorClass{}
}

type StepActivatorClass struct{}

func (*StepActivatorClass) F(f float64) float64 {
	if f > 0 {
		return 1
	}

	return 0
}

func (*StepActivatorClass) D(x float64) float64 {
	return 1
}

func CreateStepActivator() ActivatorClass {
	return &StepActivatorClass{}
}
