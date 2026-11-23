package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
)

type grepArguments struct {
	Pattern string `json:"pattern"`
	Path    string `json:"path"`
}

func HandleGrep(ctx context.Context, msg json.RawMessage) (string, error) {
	var args grepArguments

	err := json.Unmarshal(msg, &args)

	if err != nil {
		return "", fmt.Errorf("Error parsing grep arguments: %w", err)
	}

	if args.Pattern == "" {
		return "", fmt.Errorf("Pattern is missing")
	}

	if args.Path == "" {
		args.Path = "."
	}

	fileInfo, err := os.Stat(args.Path)
	if err != nil {
		return "", fmt.Errorf("File path does not exist")
	}

	if !fileInfo.IsDir() {
		return "", fmt.Errorf("File path is not a directory")
	}

	cmd := exec.CommandContext(ctx, "rg",
		"--no-config", "--color=never", "--no-heading",
		"--with-filename", "--line-number",
		"-e", args.Pattern,
		"--", args.Path,
	)
	output, err := cmd.Output()

	if err != nil {

		if errors.Is(err, exec.ErrNotFound) {
			return "", fmt.Errorf("ripgrep (rg) not found on PATH; install it or add it to PATH.")
		}

		var exitError *exec.ExitError

		if errors.As(err, &exitError) && exitError.ExitCode() == 1 {
			return "No Matches.", nil
		}

		return "", fmt.Errorf("ripgrep error: %w", err)
	}

	return string(output), nil
}
