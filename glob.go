package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/bmatcuk/doublestar/v4"
)

type globArguments struct {
	Path    string `json:"path"`
	Pattern string `json:"pattern"`
}

func HandleGlob(ctx context.Context, msg json.RawMessage) (string, error) {
	var args globArguments
	err := json.Unmarshal(msg, &args)

	if err != nil {
		return "", fmt.Errorf("Error parsing glob args: %w", err)
	}

	if args.Path == "" {
		return "", fmt.Errorf("Path is missing")
	}

	fileInfo, err := os.Stat(args.Path)
	if err != nil {
		return "", fmt.Errorf("File path does not exist")
	}

	if !fileInfo.IsDir() {
		return "", fmt.Errorf("File path is not a directory")
	}

	if !doublestar.ValidatePattern(args.Pattern) {
		return "", fmt.Errorf("Glob pattern is invalid")
	}

	fs := os.DirFS(args.Path)

	res, err := doublestar.Glob(fs, args.Pattern)

	if err != nil {
		return "", fmt.Errorf("Error during glob %w", err)
	}

	results := strings.Join(res, "\n")

	return results, nil
}
