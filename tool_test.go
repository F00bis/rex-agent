package main

import (
	"reflect"
	"testing"

	"github.com/OpenRouterTeam/go-sdk/models/components"
)

func TestLoadToolDefinitions(t *testing.T) {
	path := "./tool_test.json"

	expected := components.CreateChatFunctionToolChatFunctionToolFunction(
		components.ChatFunctionToolFunction{
			Type: components.ChatFunctionToolTypeFunction,
			Function: components.ChatFunctionToolFunctionFunction{
				Name:        "glob",
				Description: new(string("lists file names that match a given glob pattern")),
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"path": map[string]any{
							"type":        "string",
							"description": "The path of the directory to list files",
						},
						"pattern": map[string]any{
							"type":        "string",
							"description": "the glob pattern to match files against",
						},
					},
					"required":             []any{"path", "pattern"},
					"additionalProperties": false,
				},
			},
		},
	)

	actual, err := LoadToolDefinition(path)

	if err != nil {
		t.Fatalf("Error in loading tool definitions %v", err)
	}

	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("Tool definition mismatch: got %v, want %v", actual, expected)
	}
}
