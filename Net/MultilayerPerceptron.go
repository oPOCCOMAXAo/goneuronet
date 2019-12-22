package Net

import (
	"fmt"
	"github.com/opoccomaxao/goneuronet/Core"
	"strings"
)

type MultilayerPerceptron struct {
	layerc       int
	neuronLayers []*Core.NeuronLayer
	buffers      []Core.IOVector
	inc          int
	output       Core.IOVector
	input        Core.IOVector
	neurons      []*Core.Neuron
	layersLength []int
	outputLayer  *Core.NeuronLayer
	inputLayer   *Core.NeuronLayer
}

// [inputData neurons output]
func CreateMultilayerPerceptron(eachLayerLength ...int) *MultilayerPerceptron {
	lc := len(eachLayerLength)
	layers := make([]*Core.NeuronLayer, lc-1)
	buffers := make([]Core.IOVector, lc)
	activator := Core.CreateSigmoidActivator() // Core.CreateHardSigmoidActivatorClass() //
	buffers[0] = Core.CreateIOVectorByLength(eachLayerLength[0])
	for i := 0; i < lc-1; i++ {
		layer := Core.CreateNeuronLayer(eachLayerLength[i], eachLayerLength[i+1], *activator)
		buffer := Core.CreateIOVectorByLength(eachLayerLength[i+1])
		layer.ConnectToInput(buffers[i])
		layer.ConnectToOutput(buffer)
		layers[i] = layer
		buffers[i+1] = buffer
	}
	return &MultilayerPerceptron{
		layerc:       lc,
		neuronLayers: layers,
		buffers:      buffers,
		inc:          eachLayerLength[0],
		input:        buffers[0],
		output:       buffers[lc-1],
		layersLength: eachLayerLength,
		outputLayer:  layers[lc-2],
		inputLayer:   layers[0],
	}
}

func (m *MultilayerPerceptron) SetActivator(class Core.ActivatorClass) {
	for _, l := range m.neuronLayers {
		l.SetActivator(class)
	}
}

func (m *MultilayerPerceptron) InitRandom() {
	for _, n := range m.neuronLayers {
		n.InitRandom()
	}
}

func (m *MultilayerPerceptron) InitConst(c Core.NetDataType) {
	for _, n := range m.neuronLayers {
		n.InitConst(c)
	}
}

func (m *MultilayerPerceptron) assignInput(input Core.IOVector) {
	copy(m.input, input)
}

func (m *MultilayerPerceptron) evaluate() {
	for _, n := range m.neuronLayers {
		n.Evaluate()
	}
}

func (m *MultilayerPerceptron) Solve(input Core.IOVector) Core.IOVector {
	m.assignInput(input)
	m.evaluate()
	return m.output
}

func (m *MultilayerPerceptron) ToString() string {
	res := make([]string, len(m.neuronLayers))
	for i, n := range m.neuronLayers {
		res[i] = n.ToString()
	}
	return strings.Join(res, "\n\n")
}

func (m *MultilayerPerceptron) Train(samples Core.SampleArray, epochs int, maxError Core.NetDataType) {
	deltaErr := make([]Core.IOVector, m.layerc)
	last := m.layerc - 1
	for i := last; i >= 0; i-- {
		deltaErr[i] = Core.CreateIOVectorByLength(m.layersLength[i] + 1)
	}
	buffer := m.buffers[last]
	dErr := deltaErr[last]
	count := m.layersLength[last]
	var gError Core.NetDataType
	for e := 0; e < epochs; e++ {
		samples.Shuffle()
		gError = 0
		for _, s := range samples {
			m.assignInput(s.In)
			m.evaluate()
			for i := 0; i < count; i++ {
				dErr[i] = buffer[i] - s.Out[i]
				gError += Core.Abs(dErr[i])
			}
			for i := last - 1; i >= 0; i-- {
				m.neuronLayers[i].BackPropagate(deltaErr[i+1], s.Speed, deltaErr[i])
			}
		}
		fmt.Printf("#%d: Error = %f\n", e, gError)
		if gError <= maxError {
			break
		}
	}
}

func (m *MultilayerPerceptron) Export() NetState {
	layers := make([]Core.LayerState, m.layerc-1)
	conns := make([]LayerConnection, m.layerc)
	conns[0] = LayerConnection{InputId: -1, OutputId: 0}
	for i := 0; i < m.layerc-1; i++ {
		layers[i] = m.neuronLayers[i].Export(i)
		conns[i+1] = LayerConnection{InputId: i, OutputId: i + 1}
	}
	conns[m.layerc-1] = LayerConnection{InputId: m.layerc - 2, OutputId: -1}
	return NetState{
		Type:        "MultilayerPerceptron",
		Layers:      layers,
		Connections: conns,
	}
}

func ImportMultilayerPerceptron(state NetState) *MultilayerPerceptron {
	lc := len(state.Layers) + 1
	eachLayerLength := make([]int, lc)
	layers := make([]*Core.NeuronLayer, lc-1)
	buffers := make([]Core.IOVector, lc)
	eachLayerLength[0] = state.Layers[0].InputLength
	buffers[0] = Core.CreateIOVectorByLength(eachLayerLength[0])
	for i := 0; i < lc-1; i++ {
		layer := Core.ImportNeuronLayer(state.Layers[i])
		eachLayerLength[i+1] = state.Layers[i].OutputLength
		buffer := Core.CreateIOVectorByLength(eachLayerLength[i+1])
		layers[i] = layer
		buffers[i+1] = buffer
	}
	for _, conn := range state.Connections {
		if conn.OutputId > -1 {
			layers[conn.OutputId].ConnectToInput(buffers[conn.InputId+1])
		} else {
			conn.OutputId = lc - 1
		}
		if conn.InputId > -1 {
			layers[conn.InputId].ConnectToOutput(buffers[conn.OutputId])
		}
	}
	return &MultilayerPerceptron{
		layerc:       lc,
		neuronLayers: layers,
		buffers:      buffers,
		inc:          eachLayerLength[0],
		input:        buffers[0],
		output:       buffers[lc-1],
		layersLength: eachLayerLength,
		outputLayer:  layers[lc-2],
		inputLayer:   layers[0],
	}
}
