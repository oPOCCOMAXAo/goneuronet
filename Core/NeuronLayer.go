package Core

import (
	"strings"
)

type NeuronLayer struct {
	neurons   []*Neuron
	input     IOVector
	output    IOVector
	sum       IOVector
	activator ActivatorClass
	inCount   int
	outCount  int
}

type LayerState struct {
	Id           int           `json:"id"`
	Neurons      []NeuronState `json:"neurons"`
	Activator    string        `json:"activator"`
	InputLength  int           `json:"input_length"`
	OutputLength int           `json:"output_length"`
}

func CreateNeuronLayer(inputCount int, outputCount int, activator ActivatorClass) *NeuronLayer {
	if activator == nil {
		activator = *CreateLinearActivator()
	}
	res := NeuronLayer{
		neurons:   make([]*Neuron, outputCount),
		input:     CreateIOVectorByLength(inputCount),
		output:    CreateIOVectorByLength(outputCount),
		sum:       CreateIOVectorByLength(outputCount),
		inCount:   inputCount,
		outCount:  outputCount,
		activator: activator,
	}
	for i := 0; i < outputCount; i++ {
		res.neurons[i] = CreateNeuron(inputCount)
	}
	return &res
}

func (nl *NeuronLayer) SetActivator(class ActivatorClass) {
	nl.activator = class
}

func (nl *NeuronLayer) ConnectToInput(inputArray IOVector) {
	nl.input = inputArray[:]
}

func (nl *NeuronLayer) ConnectToOutput(outputArray IOVector) {
	nl.output = outputArray[:]
}

func (nl *NeuronLayer) Evaluate() {
	for i := 0; i < nl.outCount; i++ {
		n := nl.neurons[i]
		var sum NetDataType = 0
		for j := 0; j < nl.inCount; j++ {
			sum += n.Weight[j] * nl.input[j]
		}
		sum += n.Bias
		nl.sum[i] = sum
		nl.output[i] = nl.activator.F(sum)
	}
}

func (nl *NeuronLayer) EvaluateGet() IOVector {
	nl.Evaluate()
	return nl.output
}

func (nl *NeuronLayer) InitRandom() {
	for i := 0; i < nl.outCount; i++ {
		for j := 0; j < nl.inCount; j++ {
			nl.neurons[i].Weight[j] = Rand() - 0.5
		}
	}
}

func (nl *NeuronLayer) InitConst(c NetDataType) {
	for i := 0; i < nl.outCount; i++ {
		for j := 0; j < nl.inCount; j++ {
			nl.neurons[i].Weight[j] = c
		}
	}
}

func (nl *NeuronLayer) BackPropagate(deltaError IOVector, speed NetDataType, deltaErrNext IOVector) {
	for i := 0; i < nl.inCount; i++ {
		deltaErrNext[i] = 0
	}

	//mult := 1.0 / NetDataType(nl.inCount)
	for i := 0; i < nl.outCount; i++ {
		weights := nl.neurons[i].Weight
		deltaError[i] *= nl.activator.D(nl.sum[i])
		dE := deltaError[i]
		for j := 0; j < nl.inCount; j++ {
			deltaErrNext[j] += dE * weights[j]
		}
	}
	//speed *= mult
	for i := 0; i < nl.outCount; i++ {
		n := nl.neurons[i]
		e := speed * deltaError[i]
		for j := 0; j < nl.inCount; j++ {
			n.Weight[j] -= e * nl.input[j]
		}
		n.Bias -= e
	}
}

func (nl *NeuronLayer) ToString() string {
	res := make([]string, nl.outCount)
	for i := 0; i < nl.outCount; i++ {
		t := make([]string, nl.inCount+1)
		for j := 0; j < nl.inCount; j++ {
			t[j] = FloatToFixed(nl.neurons[i].Weight[j])
		}
		t[nl.inCount] = FloatToFixed(nl.neurons[i].Bias)
		res[i] = strings.Join(t, " ")
	}
	return strings.Join(res, "\n")
}

func (nl *NeuronLayer) Export(id int) LayerState {
	ns := make([]NeuronState, nl.outCount)
	for i := 0; i < nl.outCount; i++ {
		ns[i] = nl.neurons[i].Export(i)
	}
	var activator string
	switch nl.activator.(type) {
	case StepActivatorClass:
		activator = "StepActivatorClass"
	case SigmoidActivatorClass:
		activator = "SigmoidActivatorClass"
	case HardSigmoidActivatorClass:
		activator = "HardSigmoidActivatorClass"
	default:
		activator = "LinearActivatorClass"
	}
	return LayerState{
		Id:           id,
		Neurons:      ns,
		Activator:    activator,
		InputLength:  nl.inCount,
		OutputLength: nl.outCount,
	}
}

func ImportNeuronLayer(state LayerState) *NeuronLayer {
	var activator *ActivatorClass
	switch state.Activator {
	case "StepActivatorClass":
		activator = CreateStepActivator()
	case "SigmoidActivatorClass":
		activator = CreateSigmoidActivator()
	case "HardSigmoidActivatorClass":
		activator = CreateHardSigmoidActivatorClass()
	default:
		activator = CreateLinearActivator()
	}
	if activator == nil {
		activator = CreateLinearActivator()
	}
	outputCount := state.OutputLength
	inputCount := state.InputLength
	res := NeuronLayer{
		neurons:   make([]*Neuron, outputCount),
		input:     CreateIOVectorByLength(inputCount),
		output:    CreateIOVectorByLength(outputCount),
		sum:       CreateIOVectorByLength(outputCount),
		inCount:   inputCount,
		outCount:  outputCount,
		activator: *activator,
	}
	for i := 0; i < outputCount; i++ {
		res.neurons[i] = ImportNeuron(state.Neurons[i])
	}
	return &res
}
