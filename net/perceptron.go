package net

import (
	"math"

	"github.com/opoccomaxao-go/neuronet/core"
)

type Perceptron struct {
	neuronLayer *core.NeuronLayer
	input       core.IOVector
	size        int
}

func CreatePerceptron(inputCount int) *Perceptron {
	res := &Perceptron{
		neuronLayer: core.CreateNeuronLayer(inputCount, 1, core.CreateStepActivator()),
		input:       core.CreateIOVectorByLength(inputCount),
		size:        inputCount,
	}
	res.neuronLayer.ConnectToInput(res.input)

	return res
}

func (p *Perceptron) assignInput(input core.IOVector) {
	copy(p.input, input)
}

func (p *Perceptron) solve(input core.IOVector) float64 {
	p.assignInput(input)

	return p.neuronLayer.EvaluateGet()[0]
}

func (p *Perceptron) SetActivator(class core.ActivatorClass) {
	p.neuronLayer.SetActivator(class)
}

func (p *Perceptron) InitRandom() {
	p.neuronLayer.InitRandom()
}

func (p *Perceptron) InitConst(c float64) {
	p.neuronLayer.InitConst(c)
}

func (p *Perceptron) Solve(input core.IOVector) core.IOVector {
	return core.CreateIOVector(p.solve(input))
}

func (p *Perceptron) Train(
	samples core.SampleArray,
	epochs int,
	maxError float64,
) (*TrainResult, error) {
	dErr := core.CreateIOVectorByLength(1)
	dErrNext := core.CreateIOVectorByLength(p.size + 1)

	var (
		gError float64
		res    TrainResult
	)

	for epoch := 0; epoch < epochs; epoch++ {
		samples.Shuffle()

		gError = 0

		for _, s := range samples {
			res := p.solve(s.In)
			dErr[0] = res - s.Out[0]
			gError += math.Abs(dErr[0])
			p.neuronLayer.BackPropagate(dErr, s.Speed, dErrNext)
		}

		res.Errors = append(res.Errors, gError)

		if gError <= maxError {
			break
		}
	}

	return &res, nil
}

func (p *Perceptron) ToString() string {
	return p.neuronLayer.ToString()
}

func (p *Perceptron) Export() State {
	return State{
		Type:        "Perceptron",
		Layers:      []core.LayerState{p.neuronLayer.Export(0)},
		Connections: []LayerConnection{{InputID: -1, OutputID: 0}, {InputID: 0, OutputID: -1}},
	}
}

func ImportPerceptron(state State) (*Perceptron, error) {
	res := &Perceptron{
		neuronLayer: core.ImportNeuronLayer(state.Layers[0]),
		input:       core.CreateIOVectorByLength(state.Layers[0].InputLength),
		size:        state.Layers[0].InputLength,
	}
	res.neuronLayer.ConnectToInput(res.input)

	return res, nil
}
