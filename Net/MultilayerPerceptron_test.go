package Net

import (
	"fmt"
	"github.com/opoccomaxao/goneuronet/Core"
	"os"
	"testing"
)

func TestCreateMultilayerPerceptron(t *testing.T) {
	m := CreateMultilayerPerceptron(5, 3, 1)
	fmt.Printf("%#v\n", m)
	fmt.Println(m.ToString())
	state := m.Export()
	fmt.Printf("%#v\n", state)
	fmt.Printf("%s\n", Export(state))
}

func CreateXORSamples() Core.SampleArray {
	samples := make(Core.SampleArray, 0)
	samples = append(samples, Core.CreateSample(Core.CreateIOVector(0, 0), Core.CreateIOVector(0)))
	samples = append(samples, Core.CreateSample(Core.CreateIOVector(1, 0), Core.CreateIOVector(1)))
	samples = append(samples, Core.CreateSample(Core.CreateIOVector(0, 1), Core.CreateIOVector(1)))
	samples = append(samples, Core.CreateSample(Core.CreateIOVector(1, 1), Core.CreateIOVector(0)))
	return samples
}

func TestMultilayerPerceptronTrainSave(t *testing.T) {
	m := CreateMultilayerPerceptron(2, 2, 1)
	m.SetActivator(*Core.CreateHardSigmoidActivatorClass())
	m.InitRandom()
	samples := CreateXORSamples()
	var maxError Core.NetDataType = 0.00001
	m.Train(samples, Infinity, maxError)
	fmt.Printf("%v\n", m.ToString())
	for _, s := range samples {
		solved := m.Solve(s.In)
		sample := s.Out[0]
		if Core.Abs(sample-solved[0]) > maxError {
			t.Errorf("%s != %f\n", s.ToString(), solved)
		}
	}
	Save(m.Export(), "trainedNet")
}

func TestMultilayerPerceptronLoadTest(t *testing.T) {
	samples := CreateXORSamples()
	var maxError Core.NetDataType = 0.00001
	imported := Load("trainedNet")
	for _, s := range samples {
		solved := imported.Solve(s.In)
		sample := s.Out[0]
		if Core.Abs(sample-solved[0]) > maxError {
			t.Errorf("%s != %f\n", s.ToString(), solved)
		}
	}
	_ = os.Remove("trainedNet")
}
