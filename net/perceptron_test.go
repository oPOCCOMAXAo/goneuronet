package net

import (
	"testing"

	"github.com/opoccomaxao-go/neuronet/core"
)

func TestPerceptron(t *testing.T) {
	t.Parallel()

	net := CreatePerceptron(2)
	samples := core.SampleArray{
		core.CreateSample(core.CreateIOVector(0, 0), core.CreateIOVector(0)),
		core.CreateSample(core.CreateIOVector(1, 0), core.CreateIOVector(0)),
		core.CreateSample(core.CreateIOVector(0, 1), core.CreateIOVector(0)),
		core.CreateSample(core.CreateIOVector(1, 1), core.CreateIOVector(1)),
	}

	net.InitConst(0.5)
	res, _ := net.Train(samples, 100, 0)
	t.Logf("%v\n", res)
	t.Logf("%v\n", net.ToString())

	for _, s := range samples {
		solved := net.Solve(s.In)[0]
		sample := s.Out[0]

		if sample != solved {
			t.Errorf("%s != %f\n", s.ToString(), solved)
		}
	}
}
