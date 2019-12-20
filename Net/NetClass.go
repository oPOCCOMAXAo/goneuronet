package Net

import (
	"encoding/json"
	"fmt"
	"github.com/opoccomaxao/goneuronet/Core"
	"io/ioutil"
	"math"
	"os"
)

const Infinity = math.MaxInt64

type NetClass interface {
	Solve(input Core.IOVector) Core.IOVector
	Train(samples Core.SampleArray, numEpochs int, maxError Core.NetDataType)
	Export() NetState
}

type LayerConnection struct {
	InputId  int `json:"input_id"`
	OutputId int `json:"output_id"`
}

type NetState struct {
	Type        string            `json:"type"`
	Layers      []Core.LayerState `json:"layers"`
	Connections []LayerConnection `json:"connections"`
}

func Import(data []byte) NetClass {
	var state NetState
	err := json.Unmarshal(data, &state)
	if err != nil {
		printErr(err)
	}
	switch state.Type {
	case "MultilayerPerceptron":
		return ImportMultilayerPerceptron(state)
	case "Perceptron":
		return ImportPerceptron(state)
	}
	return nil
}

func Export(state NetState) []byte {
	arr, err := json.Marshal(state)
	if err != nil {
		printErr(err)
	}
	return arr
}

func Save(data NetState, fname string) {
	err := ioutil.WriteFile(fname, Export(data), os.FileMode(666))
	if err != nil {
		printErr(err)
	}
}

func Load(fname string) NetClass {
	arr, err := ioutil.ReadFile(fname)
	if err != nil {
		printErr(err)
	}
	return Import(arr)
}

func printErr(err error) {
	fmt.Printf("Error: %v\n", err)
}
