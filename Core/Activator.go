package Core

type ActivatorClass interface {
	F(NetDataType) NetDataType // forward Function
	D(NetDataType) NetDataType // backward Derivative
}

type SigmoidActivatorClass struct{}

func (s SigmoidActivatorClass) F(x NetDataType) NetDataType {
	return 1.0 / (1.0 + Exp(-x))
}

func (s SigmoidActivatorClass) D(x NetDataType) NetDataType {
	ex := Exp(-x)
	return ex / Sqr(ex+1.0)
}

func CreateSigmoidActivator() *ActivatorClass {
	var res ActivatorClass = SigmoidActivatorClass{}
	return &res
}

type HardSigmoidActivatorClass struct{}

func (s HardSigmoidActivatorClass) F(x NetDataType) NetDataType {
	if x < -2.5 {
		return 0
	} else if x > 2.5 {
		return 1
	} else {
		return 0.2*x + 0.5
	}
}

func (s HardSigmoidActivatorClass) D(x NetDataType) NetDataType {
	if x < -2.5 {
		return 0
	} else if x > 2.5 {
		return 0
	} else {
		return 0.2
	}
}

func CreateHardSigmoidActivatorClass() *ActivatorClass {
	var res ActivatorClass = HardSigmoidActivatorClass{}
	return &res
}

type LinearActivatorClass struct{}

func (r LinearActivatorClass) F(x NetDataType) NetDataType {
	return x
}

func (r LinearActivatorClass) D(x NetDataType) NetDataType {
	return 1
}

func CreateLinearActivator() *ActivatorClass {
	var res ActivatorClass = LinearActivatorClass{}
	return &res
}

type StepActivatorClass struct{}

func (s StepActivatorClass) F(f NetDataType) NetDataType {
	if f > 0 {
		return 1
	} else {
		return 0
	}
}

func (s StepActivatorClass) D(x NetDataType) NetDataType {
	return 1
}

func CreateStepActivator() *ActivatorClass {
	var res ActivatorClass = StepActivatorClass{}
	return &res
}
