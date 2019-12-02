package Core

type Neuron struct {
	Weight IOVector
}

type NeuronState struct {
	Id      int      `json:"id"`
	Weights IOVector `json:"weights"`
}

func CreateNeuron(size int) *Neuron {
	return &Neuron{
		Weight: CreateIOVectorByLength(size),
	}
}

func (n *Neuron) Export(id int) NeuronState {
	return NeuronState{
		Id:      id,
		Weights: n.Weight,
	}
}

func ImportNeuron(state NeuronState) *Neuron {
	return &Neuron{Weight: state.Weights}
}
