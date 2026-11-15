package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/OpenRouterTeam/go-sdk/models/components"
)

type Tool struct {
	Definition components.ChatFunctionTool
	Execute    func(context.Context, json.RawMessage) (string, error)
}

func NewTool(path string, execute func(context.Context, json.RawMessage) (string, error)) (Tool, error) {
	definition, err := LoadToolDefinition(path)

	if err != nil {
		return Tool{}, err
	}

	return Tool{definition, execute}, nil
}

func LoadToolDefinition(path string) (components.ChatFunctionTool, error) {
	data, err := os.ReadFile(path)

	if err != nil {
		return components.ChatFunctionTool{}, err
	}

	var definition components.ChatFunctionTool
	if err := json.Unmarshal(data, &definition); err != nil {
		return components.ChatFunctionTool{}, fmt.Errorf("decode tool definition error: %w", err)
	}

	return definition, nil
}
