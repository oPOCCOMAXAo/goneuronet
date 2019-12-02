package Net

import (
	"fmt"
	"testing"
)

func TestParenthesis(t *testing.T) {
	var a NetClass = CreateMultilayerPerceptron(2, 1)
	fmt.Printf("%#v\n", a)
	a = CreatePerceptron(2)
	fmt.Printf("%#v\n", a)
}
