package Net

import (
	"fmt"
	"github.com/opoccomaxao/goneuronet/Core"
	"testing"
)

func TestPerceptron(t *testing.T) {
	p := CreatePerceptron(2)
	samples := make([]Core.Sample, 0)
	samples = append(samples, Core.CreateSample(Core.CreateIOVector(0, 0), Core.CreateIOVector(0)))
	samples = append(samples, Core.CreateSample(Core.CreateIOVector(1, 0), Core.CreateIOVector(0)))
	samples = append(samples, Core.CreateSample(Core.CreateIOVector(0, 1), Core.CreateIOVector(0)))
	samples = append(samples, Core.CreateSample(Core.CreateIOVector(1, 1), Core.CreateIOVector(1)))
	p.InitConst(0.5)
	p.Train(samples, 0.1, 100, 0)
	fmt.Printf("%v\n", p.ToString())
	for _, s := range samples {
		solved := p.Solve(s.In)[0]
		sample := s.Out[0]
		if sample != solved {
			t.Errorf("%s != %f\n", s.ToString(), solved)
		}
	}
}
