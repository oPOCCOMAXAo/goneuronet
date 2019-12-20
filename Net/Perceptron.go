package Net

import (
	"fmt"
	"github.com/opoccomaxao/goneuronet/Core"
)

type Perceptron struct {
	neuronLayer *Core.NeuronLayer
	input       Core.IOVector
	size        int
}

func CreatePerceptron(inputCount int) *Perceptron {
	res := &Perceptron{
		neuronLayer: Core.CreateNeuronLayer(inputCount, 1, Core.CreateStepActivator()),
		input:       Core.CreateIOVectorByLength(inputCount),
		size:        inputCount,
	}
	res.neuronLayer.ConnectToInput(res.input)
	return res
}

func (p *Perceptron) assignInput(input Core.IOVector) {
	copy(p.input, input)
}

func (p *Perceptron) solve(input Core.IOVector) Core.NetDataType {
	p.assignInput(input)
	return *p.neuronLayer.EvaluateGet()[0]
}

func (p *Perceptron) InitRandom() {
	p.neuronLayer.InitRandom()
}

func (p *Perceptron) InitConst(c Core.NetDataType) {
	p.neuronLayer.InitConst(c)
}

func (p *Perceptron) Solve(input Core.IOVector) Core.IOVector {
	return Core.CreateIOVector(p.solve(input))
}

func (p *Perceptron) Train(samples Core.SampleArray, epochs int, maxError Core.NetDataType) {
	dErr := Core.CreateIOVectorByLength(1)
	dErrNext := Core.CreateIOVectorByLength(p.size + 1)
	var gError Core.NetDataType
	for e := 0; e < epochs; e++ {
		samples.Shuffle()
		gError = 0
		for _, s := range samples {
			res := p.solve(s.In)
			dErr[0] = res - s.Out[0]
			gError += Core.Abs(dErr[0])
			p.neuronLayer.BackPropagate(dErr, s.Speed, dErrNext)
		}
		fmt.Printf("#%d: Error = %f\n", e, gError)
		if gError <= maxError {
			break
		}
	}
}

func (p *Perceptron) ToString() string {
	return p.neuronLayer.ToString()
}

func (p *Perceptron) Export() NetState {
	return NetState{
		Type:        "Perceptron",
		Layers:      []Core.LayerState{p.neuronLayer.Export(0)},
		Connections: []LayerConnection{{InputId: -1, OutputId: 0}, {InputId: 0, OutputId: -1}},
	}
}

func ImportPerceptron(state NetState) *Perceptron {
	res := &Perceptron{
		neuronLayer: Core.ImportNeuronLayer(state.Layers[0]),
		input:       Core.CreateIOVectorByLength(state.Layers[0].InputLength),
		size:        state.Layers[0].InputLength,
	}
	res.neuronLayer.ConnectToInput(res.input)
	return res
}
