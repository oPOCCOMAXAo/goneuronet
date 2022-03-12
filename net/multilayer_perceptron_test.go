package net

import (
	_ "embed"
	"os"
	"testing"

	"github.com/opoccomaxao/goneuronet/core"
	"github.com/stretchr/testify/require"
)

//go:embed testdata/mpl-5-3-1.json
var mlp531State string

func TestCreateMultilayerPerceptron(t *testing.T) {
	t.Parallel()

	mlp531 := CreateMultilayerPerceptron(5, 3, 1)
	require.Equal(t, ` 0.000  0.000  0.000  0.000  0.000  0.000
 0.000  0.000  0.000  0.000  0.000  0.000
 0.000  0.000  0.000  0.000  0.000  0.000

 0.000  0.000  0.000  0.000`, mlp531.String())

	data, err := Export(mlp531.Export())
	require.NoError(t, err)
	require.JSONEq(t, mlp531State, string(data))
}

func CreateXORSamples() core.SampleArray {
	return core.SampleArray{
		core.CreateSample(core.CreateIOVector(0, 0), core.CreateIOVector(0)),
		core.CreateSample(core.CreateIOVector(1, 0), core.CreateIOVector(1)),
		core.CreateSample(core.CreateIOVector(0, 1), core.CreateIOVector(1)),
		core.CreateSample(core.CreateIOVector(1, 1), core.CreateIOVector(0)),
	}
}

func TestMultilayerPerceptronTrainSaveLoad(t *testing.T) {
	t.Parallel()

	net := CreateMultilayerPerceptron(2, 2, 1)
	net.SetActivator(core.CreateHardSigmoidActivatorClass())
	net.InitRandom()

	var (
		samples  = CreateXORSamples()
		maxError = core.NetDataType(0.00001)
	)

	res, _ := net.Train(samples, Infinity, maxError)
	t.Logf("Errors:%v\n", res.Errors)
	t.Logf("%s\n", net.String())

	for _, s := range samples {
		solved := net.Solve(s.In)
		sample := s.Out[0]

		if core.Abs(sample-solved[0]) > maxError {
			t.Errorf("%s != %f\n", s.ToString(), solved)
		}
	}

	require.NoError(t, Save(net.Export(), "trainedNet"))

	imported, err := Load("trainedNet")
	require.NoError(t, err)

	for _, s := range samples {
		solved := imported.Solve(s.In)
		sample := s.Out[0]

		if core.Abs(sample-solved[0]) > maxError {
			t.Errorf("%s != %f\n", s.ToString(), solved)
		}
	}

	_ = os.Remove("trainedNet")
}
