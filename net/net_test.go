package net

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestParenthesis(t *testing.T) {
	t.Parallel()

	data, err := json.Marshal(CreateMultilayerPerceptron(2, 1))
	assert.NoError(t, err)
	assert.JSONEq(t, `{}`, string(data))

	data, err = json.Marshal(CreatePerceptron(2))
	assert.NoError(t, err)
	assert.JSONEq(t, `{}`, string(data))
}
