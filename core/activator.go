package core

type ActivatorClass interface {
	F(NetDataType) NetDataType // F forward Function
	D(NetDataType) NetDataType // D backward Derivative
}

type SigmoidActivatorClass struct{}

func (s SigmoidActivatorClass) F(x NetDataType) NetDataType {
	return 1.0 / (1.0 + Exp(-x))
}

func (s SigmoidActivatorClass) D(x NetDataType) NetDataType {
	exp := Exp(-x)

	return exp / Sqr(exp+1.0)
}

func CreateSigmoidActivator() ActivatorClass {
	return &SigmoidActivatorClass{}
}

type HardSigmoidActivatorClass struct{}

func (HardSigmoidActivatorClass) F(value NetDataType) NetDataType {
	if value < -2.5 {
		return 0
	}

	if value > 2.5 {
		return 1
	}

	return 0.2*value + 0.5
}

func (HardSigmoidActivatorClass) D(x NetDataType) NetDataType {
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

func (LinearActivatorClass) F(x NetDataType) NetDataType {
	return x
}

func (LinearActivatorClass) D(x NetDataType) NetDataType {
	return 1
}

func CreateLinearActivator() ActivatorClass {
	return &LinearActivatorClass{}
}

type StepActivatorClass struct{}

func (StepActivatorClass) F(f NetDataType) NetDataType {
	if f > 0 {
		return 1
	}

	return 0
}

func (StepActivatorClass) D(x NetDataType) NetDataType {
	return 1
}

func CreateStepActivator() ActivatorClass {
	return &StepActivatorClass{}
}
