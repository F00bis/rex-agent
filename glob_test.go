package main

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestHandleGlobMatches(t *testing.T) {
	root := newGlobTestDir(t)
	tests := []struct {
		name      string
		path      string
		pattern   string
		want      string
		unordered bool
	}{
		{name: "exact file", path: root, pattern: "README.md", want: "README.md"},
		{name: "files and directory with matching extension", path: root, pattern: "*.go", want: "a.go\ngenerated.go\nz.go"},
		{
			name: "recursive files and directories", path: root, pattern: "**/*.go",
			want: "a.go\ngenerated.go\nsrc/b.go\nsrc/deep/c.go\nz.go", unordered: true,
		},
		{name: "scoped files", path: root, pattern: "src/*.go", want: "src/b.go"},
		{name: "no matches", path: root, pattern: "*.rs", want: ""},
		{name: "empty root directory", path: filepath.Join(root, "empty"), pattern: "*", want: ""},
		{name: "exact directory excludes descendants", path: root, pattern: "src", want: "src"},
		{name: "mixed nested entries", path: root, pattern: "src/*", want: "src/b.go\nsrc/deep"},
		{name: "empty directory matches by name", path: root, pattern: "empty", want: "empty"},
		{name: "search inside empty directory", path: root, pattern: "empty/*", want: ""},
		{name: "hidden file", path: root, pattern: "*.txt", want: ".hidden.txt"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := HandleGlob(t.Context(), globTestArgs(t, tt.path, tt.pattern))
			if err != nil {
				t.Fatal(err)
			}
			if strings.HasSuffix(got, "\n") {
				t.Errorf("output = %q, want no trailing newline", got)
			}
			if tt.unordered {
				// Recursive traversal order is not part of the handler's contract.
				paths := strings.Split(got, "\n")
				slices.Sort(paths)
				got = strings.Join(paths, "\n")
			}
			if got != tt.want {
				t.Errorf("output = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestHandleGlobErrors(t *testing.T) {
	root := newGlobTestDir(t)
	wrongPathType, err := json.Marshal(map[string]any{"path": 42, "pattern": "*"})
	if err != nil {
		t.Fatal(err)
	}
	wrongPatternType, err := json.Marshal(map[string]any{"path": root, "pattern": false})
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name        string
		args        json.RawMessage
		want        string
		syntaxError bool
		typeError   bool
	}{
		{name: "malformed JSON", args: json.RawMessage(`{"path":`), want: "Error parsing glob args", syntaxError: true},
		{name: "wrong path type", args: wrongPathType, want: "Error parsing glob args", typeError: true},
		{name: "wrong pattern type", args: wrongPatternType, want: "Error parsing glob args", typeError: true},
		{name: "missing path", args: json.RawMessage(`{"pattern":"*"}`), want: "Path is missing"},
		{name: "empty path", args: globTestArgs(t, "", "*"), want: "Path is missing"},
		{name: "nonexistent root", args: globTestArgs(t, filepath.Join(root, "missing"), "*"), want: "File path does not exist"},
		{name: "regular file root", args: globTestArgs(t, filepath.Join(root, "README.md"), "*"), want: "File path is not a directory"},
		{name: "unclosed character class", args: globTestArgs(t, root, "["), want: "Glob pattern is invalid"},
		{name: "unclosed alternatives", args: globTestArgs(t, root, "{a,b"), want: "Glob pattern is invalid"},
		{name: "trailing escape", args: globTestArgs(t, root, `file\`), want: "Glob pattern is invalid"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := HandleGlob(t.Context(), tt.args)
			if got != "" {
				t.Errorf("output = %q, want empty output on error", got)
			}
			if err == nil {
				t.Fatalf("error = nil, want %q", tt.want)
			}
			if tt.syntaxError || tt.typeError {
				if !strings.Contains(err.Error(), tt.want) {
					t.Errorf("error = %v, want containing %q", err, tt.want)
				}
			} else if err.Error() != tt.want {
				t.Errorf("error = %v, want %q", err, tt.want)
			}
			if tt.syntaxError {
				var cause *json.SyntaxError
				if !errors.As(err, &cause) {
					t.Errorf("error = %v, want wrapping *json.SyntaxError", err)
				}
			}
			if tt.typeError {
				var cause *json.UnmarshalTypeError
				if !errors.As(err, &cause) {
					t.Errorf("error = %v, want wrapping *json.UnmarshalTypeError", err)
				}
			}
		})
	}
}

func newGlobTestDir(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	for _, dir := range []string{"src/deep", "empty", "generated.go"} {
		if err := os.MkdirAll(filepath.Join(root, filepath.FromSlash(dir)), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	for _, name := range []string{"a.go", "z.go", "README.md", ".hidden.txt", "src/b.go", "src/deep/c.go"} {
		if err := os.WriteFile(filepath.Join(root, filepath.FromSlash(name)), nil, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

func globTestArgs(t *testing.T, path, pattern string) json.RawMessage {
	t.Helper()
	args, err := json.Marshal(globArguments{Path: path, Pattern: pattern})
	if err != nil {
		t.Fatal(err)
	}
	return args
}
