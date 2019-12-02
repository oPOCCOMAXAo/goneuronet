package Core

import (
	"strings"
)

type NeuronLayer struct {
	neurons   []*Neuron
	input     []*NetDataType
	output    []*NetDataType
	sum       IOVector
	activator *ActivatorClass
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

var oneMainType NetDataType = 1
var zeroMainType NetDataType = 0

func CreateNeuronLayer(inputCount int, outputCount int, activator *ActivatorClass) *NeuronLayer {
	if activator == nil {
		activator = &LinearActivator
	}
	res := NeuronLayer{
		neurons:   make([]*Neuron, outputCount),
		input:     make([]*NetDataType, inputCount+1),
		output:    make([]*NetDataType, outputCount),
		sum:       CreateIOVectorByLength(outputCount),
		inCount:   inputCount + 1,
		outCount:  outputCount,
		activator: activator,
	}
	for i := 0; i < inputCount; i++ {
		res.input[i] = new(NetDataType)
	}
	res.input[inputCount] = &oneMainType
	for i := 0; i < outputCount; i++ {
		res.output[i] = new(NetDataType)
		res.neurons[i] = CreateNeuron(inputCount + 1)
	}
	return &res
}

func (nl *NeuronLayer) ConnectToInput(inputArray IOVector) {
	max := nl.inCount
	l := len(inputArray)
	if l < max {
		max = l
	}
	for i := 0; i < max; i++ {
		nl.input[i] = &inputArray[i]
	}
}

func (nl *NeuronLayer) ConnectToOutput(outputArray IOVector) {
	max := nl.outCount
	l := len(outputArray)
	if l < max {
		max = l
	}
	for i := 0; i < max; i++ {
		nl.output[i] = &outputArray[i]
	}
}

func (nl *NeuronLayer) Evaluate() {
	for i := 0; i < nl.outCount; i++ {
		weights := nl.neurons[i].Weight
		var sum NetDataType = 0
		for j := 0; j < nl.inCount; j++ {
			sum += weights[j] * *nl.input[j]
		}
		nl.sum[i] = sum
		*nl.output[i] = (*nl.activator).F(sum)
	}
}

func (nl *NeuronLayer) EvaluateGet() []*NetDataType {
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
		deltaError[i] *= (*nl.activator).D(nl.sum[i])
		dE := deltaError[i]
		for j := 0; j < nl.inCount; j++ {
			deltaErrNext[j] += dE * weights[j]
		}
	}
	//speed *= mult
	for i := 0; i < nl.outCount; i++ {
		weights := nl.neurons[i].Weight
		e := speed * deltaError[i]
		for j := 0; j < nl.inCount; j++ {
			weights[j] -= e * *nl.input[j]
		}
	}
}

func (nl *NeuronLayer) ToString() string {
	res := make([]string, nl.outCount)
	for i := 0; i < nl.outCount; i++ {
		t := make([]string, nl.inCount)
		for j := 0; j < nl.inCount; j++ {
			t[j] = FloatToFixed(nl.neurons[i].Weight[j])
		}
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
	switch (*nl.activator).(type) {
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
		InputLength:  nl.inCount - 1,
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
		activator = &LinearActivator
	}
	outputCount := state.OutputLength
	inputCount := state.InputLength
	res := NeuronLayer{
		neurons:   make([]*Neuron, outputCount),
		input:     make([]*NetDataType, inputCount+1),
		output:    make([]*NetDataType, outputCount),
		sum:       CreateIOVectorByLength(outputCount),
		inCount:   inputCount + 1,
		outCount:  outputCount,
		activator: activator,
	}
	for i := 0; i < inputCount; i++ {
		res.input[i] = new(NetDataType)
	}
	res.input[inputCount] = &oneMainType
	for i := 0; i < outputCount; i++ {
		res.output[i] = new(NetDataType)
		res.neurons[i] = ImportNeuron(state.Neurons[i])
	}
	return &res
}
