package core

import (
	"fmt"
	"math/rand"
	"time"
)

type IOVector []float64

type Sample struct {
	In    IOVector
	Out   IOVector
	Speed float64
}

func CreateSample(input IOVector, output IOVector) *Sample {
	return &Sample{In: input, Out: output, Speed: 1}
}

func CreateSampleWithSpeed(input IOVector, output IOVector, speed float64) *Sample {
	return &Sample{In: input, Out: output, Speed: speed}
}

func CreateIOVector(array ...float64) IOVector {
	return array
}

func CreateIOVectorByLength(length int) IOVector {
	return make(IOVector, length)
}

func (s *Sample) ToString() string {
	return fmt.Sprintf("f(%v)->%f", s.In, s.Out)
}

type SampleArray []*Sample

func (s SampleArray) Shuffle() {
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(s), func(i, j int) { s[i], s[j] = s[j], s[i] })
}
