package net

import (
	"encoding/json"
	"io/ioutil"
	"math"
	"os"

	"github.com/opoccomaxao-go/neuronet/core"
	"github.com/pkg/errors"
)

const Infinity = math.MaxInt64

type Net interface {
	Solve(input core.IOVector) core.IOVector
	SetActivator(class core.ActivatorClass)
	Train(samples core.SampleArray, numEpochs int, maxError float64) (*TrainResult, error)
	Export() State
}

type TrainResult struct {
	Errors []float64
}

type LayerConnection struct {
	InputID  int `json:"input_id"`
	OutputID int `json:"output_id"`
}

type State struct {
	Type        string            `json:"type"`
	Layers      []core.LayerState `json:"layers"`
	Connections []LayerConnection `json:"connections"`
}

func Import(data []byte) (Net, error) {
	var state State

	err := json.Unmarshal(data, &state)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	switch state.Type {
	case "MultilayerPerceptron":
		return ImportMultilayerPerceptron(state)
	case "Perceptron":
		return ImportPerceptron(state)
	}

	return nil, errors.WithStack(ErrTypeNotSupported)
}

func Export(state State) ([]byte, error) {
	res, err := json.Marshal(state)

	return res, errors.WithStack(err)
}

func Save(data State, fname string) error {
	bytes, err := Export(data)
	if err != nil {
		return errors.WithStack(err)
	}

	err = ioutil.WriteFile(fname, bytes, os.FileMode(0o666))

	return errors.WithStack(err)
}

func Load(fname string) (Net, error) {
	arr, err := ioutil.ReadFile(fname)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	return Import(arr)
}
