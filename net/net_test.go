package net

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestParenthesis(t *testing.T) {
	t.Parallel()

	data, _ := json.Marshal(CreateMultilayerPerceptron(2, 1))
	assert.JSONEq(t, `{}`, string(data))

	data, _ = json.Marshal(CreatePerceptron(2))
	assert.JSONEq(t, `{}`, string(data))
}
