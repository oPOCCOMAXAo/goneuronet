package net

import (
	"math"
	"strings"

	"github.com/opoccomaxao-go/neuronet/core"
)

type MultilayerPerceptron struct {
	layerc       int
	neuronLayers []*core.NeuronLayer
	buffers      []core.IOVector
	inc          int
	output       core.IOVector
	input        core.IOVector
	layersLength []int
	outputLayer  *core.NeuronLayer
	inputLayer   *core.NeuronLayer
}

func CreateMultilayerPerceptron(eachLayerLength ...int) *MultilayerPerceptron {
	layersCount := len(eachLayerLength)
	layers := make([]*core.NeuronLayer, layersCount-1)
	buffers := make([]core.IOVector, layersCount)
	activator := core.CreateSigmoidActivator() // Core.CreateHardSigmoidActivatorClass() //
	buffers[0] = core.CreateIOVectorByLength(eachLayerLength[0])

	for layerIdx := 0; layerIdx < layersCount-1; layerIdx++ {
		layer := core.CreateNeuronLayer(eachLayerLength[layerIdx], eachLayerLength[layerIdx+1], activator)
		buffer := core.CreateIOVectorByLength(eachLayerLength[layerIdx+1])
		layer.ConnectToInput(buffers[layerIdx])
		layer.ConnectToOutput(buffer)
		layers[layerIdx] = layer
		buffers[layerIdx+1] = buffer
	}

	return &MultilayerPerceptron{
		layerc:       layersCount,
		neuronLayers: layers,
		buffers:      buffers,
		inc:          eachLayerLength[0],
		input:        buffers[0],
		output:       buffers[layersCount-1],
		layersLength: eachLayerLength,
		outputLayer:  layers[layersCount-2],
		inputLayer:   layers[0],
	}
}

func (m *MultilayerPerceptron) SetActivator(class core.ActivatorClass) {
	for _, l := range m.neuronLayers {
		l.SetActivator(class)
	}
}

func (m *MultilayerPerceptron) InitRandom() {
	for _, n := range m.neuronLayers {
		n.InitRandom()
	}
}

func (m *MultilayerPerceptron) InitConst(c float64) {
	for _, n := range m.neuronLayers {
		n.InitConst(c)
	}
}

func (m *MultilayerPerceptron) assignInput(input core.IOVector) {
	copy(m.input, input)
}

func (m *MultilayerPerceptron) evaluate() {
	for _, n := range m.neuronLayers {
		n.Evaluate()
	}
}

func (m *MultilayerPerceptron) Solve(input core.IOVector) core.IOVector {
	m.assignInput(input)
	m.evaluate()

	return m.output
}

func (m *MultilayerPerceptron) String() string {
	res := make([]string, len(m.neuronLayers))
	for i, n := range m.neuronLayers {
		res[i] = n.ToString()
	}

	return strings.Join(res, "\n\n")
}

func (m *MultilayerPerceptron) Train(
	samples core.SampleArray,
	epochs int,
	maxError float64,
) (*TrainResult, error) {
	deltaErr := make([]core.IOVector, m.layerc)
	last := m.layerc - 1

	for i := last; i >= 0; i-- {
		deltaErr[i] = core.CreateIOVectorByLength(m.layersLength[i] + 1)
	}

	buffer := m.buffers[last]
	dErr := deltaErr[last]
	count := m.layersLength[last]

	var (
		gError float64
		res    TrainResult
	)

	for epoch := 0; epoch < epochs; epoch++ {
		samples.Shuffle()

		gError = 0

		for _, sample := range samples {
			m.assignInput(sample.In)
			m.evaluate()

			for i := 0; i < count; i++ {
				dErr[i] = buffer[i] - sample.Out[i]
				gError += math.Abs(dErr[i])
			}

			for i := last - 1; i >= 0; i-- {
				m.neuronLayers[i].BackPropagate(deltaErr[i+1], sample.Speed, deltaErr[i])
			}
		}

		res.Errors = append(res.Errors, gError)

		if gError <= maxError {
			break
		}
	}

	return &res, nil
}

func (m *MultilayerPerceptron) Export() State {
	layers := make([]core.LayerState, m.layerc-1)
	conns := make([]LayerConnection, m.layerc)
	conns[0] = LayerConnection{InputID: -1, OutputID: 0}

	for i := 0; i < m.layerc-1; i++ {
		layers[i] = m.neuronLayers[i].Export(i)
		conns[i+1] = LayerConnection{InputID: i, OutputID: i + 1}
	}

	conns[m.layerc-1] = LayerConnection{InputID: m.layerc - 2, OutputID: -1}

	return State{
		Type:        "MultilayerPerceptron",
		Layers:      layers,
		Connections: conns,
	}
}

func ImportMultilayerPerceptron(state State) (*MultilayerPerceptron, error) {
	layersCount := len(state.Layers) + 1
	eachLayerLength := make([]int, layersCount)
	layers := make([]*core.NeuronLayer, layersCount-1)
	buffers := make([]core.IOVector, layersCount)
	eachLayerLength[0] = state.Layers[0].InputLength
	buffers[0] = core.CreateIOVectorByLength(eachLayerLength[0])

	for i := 0; i < layersCount-1; i++ {
		layer := core.ImportNeuronLayer(state.Layers[i])
		eachLayerLength[i+1] = state.Layers[i].OutputLength
		buffer := core.CreateIOVectorByLength(eachLayerLength[i+1])
		layers[i] = layer
		buffers[i+1] = buffer
	}

	for _, conn := range state.Connections {
		if conn.OutputID > -1 {
			layers[conn.OutputID].ConnectToInput(buffers[conn.InputID+1])
		} else {
			conn.OutputID = layersCount - 1
		}

		if conn.InputID > -1 {
			layers[conn.InputID].ConnectToOutput(buffers[conn.OutputID])
		}
	}

	return &MultilayerPerceptron{
		layerc:       layersCount,
		neuronLayers: layers,
		buffers:      buffers,
		inc:          eachLayerLength[0],
		input:        buffers[0],
		output:       buffers[layersCount-1],
		layersLength: eachLayerLength,
		outputLayer:  layers[layersCount-2],
		inputLayer:   layers[0],
	}, nil
}
