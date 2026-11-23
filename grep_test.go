package main

import (
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestHandleGrepMatches(t *testing.T) {
	newGrepTestWorkspace(t)

	got, err := HandleGrep(t.Context(), json.RawMessage(`{"pattern":"^haystack [0-9]+$"}`))
	if err != nil {
		t.Fatal(err)
	}
	want := "./sample.txt:2:haystack 42\n./sample.txt:4:haystack 7\n"
	if got != want {
		t.Errorf("output = %q, want %q", got, want)
	}
}

func TestHandleGrepMatchesInSubdirectory(t *testing.T) {
	newGrepTestWorkspace(t)
	if err := os.Mkdir("src", 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join("src", "nested.txt")
	if err := os.WriteFile(path, []byte("haystack 5\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := HandleGrep(t.Context(), json.RawMessage(`{"pattern":"^haystack [0-9]+$","path":"src"}`))
	if err != nil {
		t.Fatal(err)
	}
	want := path + ":1:haystack 5\n"
	if got != want {
		t.Errorf("output = %q, want %q", got, want)
	}
}

func TestHandleGrepMissingRipgrep(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	t.Chdir(t.TempDir())

	got, err := HandleGrep(t.Context(), json.RawMessage(`{"pattern":"needle"}`))
	if err == nil || !strings.Contains(err.Error(), "ripgrep (rg) not found on PATH") {
		t.Fatalf("error = %v, want a missing ripgrep diagnostic", err)
	}
	if got != "" {
		t.Errorf("output = %q, want empty output on error", got)
	}
}

func TestHandleGrepNoMatches(t *testing.T) {
	newGrepTestWorkspace(t)

	got, err := HandleGrep(t.Context(), json.RawMessage(`{"pattern":"needle"}`))
	if err != nil {
		t.Fatal(err)
	}
	if got != "No Matches." {
		t.Errorf("output = %q, want %q", got, "No Matches.")
	}
}

func TestHandleGrepInvalidRegex(t *testing.T) {
	newGrepTestWorkspace(t)

	got, err := HandleGrep(t.Context(), json.RawMessage(`{"pattern":"["}`))
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) || exitErr.ExitCode() != 2 {
		t.Fatalf("error = %v, want ripgrep exit code 2 for invalid regex", err)
	}
	if got != "" {
		t.Errorf("output = %q, want empty output on error", got)
	}
}

func newGrepTestWorkspace(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("rg"); err != nil {
		t.Skipf("ripgrep required: %v", err)
	}
	t.Chdir(t.TempDir())
	if err := os.WriteFile("sample.txt", []byte("haystack\nhaystack 42\nother text\nhaystack 7\n"), 0o644); err != nil {
		t.Fatal(err)
	}
}
