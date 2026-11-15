package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"reflect"
	"strings"
	"testing"

	openrouter "github.com/OpenRouterTeam/go-sdk"
	"github.com/OpenRouterTeam/go-sdk/models/components"
	"github.com/OpenRouterTeam/go-sdk/retry"
)

func TestAgentRunInferenceText(t *testing.T) {
	for _, withCallback := range []bool{false, true} {
		t.Run(fmt.Sprintf("callback=%t", withCallback), func(t *testing.T) {
			nullContent := components.ChatStreamDelta{}
			nullContent.Content.Set(nil)
			ignored := agentTestTextDelta("ignored")
			ignored.ToolCalls = []components.ChatStreamToolCall{agentTestCallDelta(0, "ignored", "ignored", "{}")}
			agent, client := newAgentTestClient(t, agentTestStream(t,
				agentTestChunk(components.ChatStreamDelta{Role: new(components.ChatStreamDeltaRoleAssistant)}),
				agentTestChunk(agentTestTextDelta("Hello, ")),
				agentTestChunk(nullContent),
				components.ChatStreamChunk{Choices: []components.ChatStreamChoice{}},
				components.ChatStreamChunk{Choices: []components.ChatStreamChoice{
					{Index: 1, Delta: ignored},
					{Index: 0, Delta: agentTestTextDelta("世界")},
				}},
				agentTestChunk(agentTestTextDelta("")),
			))
			tool := agentTestTool("lookup", nil)
			agent.AddTools([]Tool{tool})
			conversation := []components.ChatMessages{agentTestUser("Hello")}
			type contextKey struct{}
			ctx := context.WithValue(t.Context(), contextKey{}, "request context")
			var deltas []string
			var callback func(string)
			if withCallback {
				callback = func(delta string) { deltas = append(deltas, delta) }
			}

			message, err := agent.RunInference(ctx, conversation, callback)
			if err != nil {
				t.Fatal(err)
			}
			assertAgentJSON(t, message, agentTestAssistant("Hello, 世界"))
			if withCallback && !reflect.DeepEqual(deltas, []string{"Hello, ", "世界", ""}) {
				t.Errorf("deltas = %q, want text fragments in order", deltas)
			}
			request := client.requests[0]
			if request.Model != "test-model" || !request.Stream {
				t.Errorf("model = %q, stream = %t; want test-model and true", request.Model, request.Stream)
			}
			if request.ctx.Value(contextKey{}) != "request context" {
				t.Error("request lost the caller's context value")
			}
			assertAgentJSON(t, request.Messages, conversation)
			assertAgentJSON(t, request.Tools, []components.ChatFunctionTool{tool.Definition})
		})
	}
}

func TestAgentRunInferenceInterleavedToolCalls(t *testing.T) {
	first := agentTestTextDelta("Checking")
	first.ToolCalls = []components.ChatStreamToolCall{{Index: 7, ID: new("call-seven")}}
	agent, _ := newAgentTestClient(t, agentTestStream(t,
		agentTestChunk(first),
		agentTestToolChunk(agentTestCallDelta(2, "call-two", "read", "{")),
		agentTestToolChunk(components.ChatStreamToolCall{Index: 7, Function: &components.ChatStreamToolCallFunction{
			Name: new("g"), Arguments: new(`{"path":`),
		}}),
		agentTestToolChunk(components.ChatStreamToolCall{Index: 2, Function: &components.ChatStreamToolCallFunction{
			Name: new("_file"), Arguments: new(`"path":"notes.txt"}`),
		}}),
		agentTestToolChunk(components.ChatStreamToolCall{Index: 7, Function: &components.ChatStreamToolCallFunction{
			Name: new("lob"), Arguments: new(`"."}`),
		}}),
	))

	message, err := agent.RunInference(t.Context(), []components.ChatMessages{agentTestUser("Check files")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := agentTestAssistant("Checking")
	want.ToolCalls = []components.ChatToolCall{
		agentTestCall("call-seven", "glob", `{"path":"."}`),
		agentTestCall("call-two", "read_file", `{"path":"notes.txt"}`),
	}
	assertAgentJSON(t, message, want)
}

func TestAgentRunInferenceWithoutText(t *testing.T) {
	for _, name := range []string{"empty stream", "role only", "tool only"} {
		t.Run(name, func(t *testing.T) {
			var chunks []components.ChatStreamChunk
			want := components.ChatAssistantMessage{Role: components.ChatAssistantMessageRoleAssistant}
			switch name {
			case "role only":
				chunks = append(chunks, agentTestChunk(components.ChatStreamDelta{Role: new(components.ChatStreamDeltaRoleAssistant)}))
			case "tool only":
				chunks = append(chunks, agentTestToolChunk(agentTestCallDelta(0, "call", "lookup", "{}")))
				want.ToolCalls = []components.ChatToolCall{agentTestCall("call", "lookup", "{}")}
			}
			agent, _ := newAgentTestClient(t, agentTestStream(t, chunks...))

			message, err := agent.RunInference(t.Context(), []components.ChatMessages{agentTestUser("Hi")}, func(delta string) {
				t.Errorf("unexpected text delta %q", delta)
			})
			if err != nil {
				t.Fatal(err)
			}
			if _, set := message.Content.Get(); set {
				t.Error("content is set for a response without text")
			}
			assertAgentJSON(t, message, want)
		})
	}
}

func TestAgentRunInferenceErrors(t *testing.T) {
	transportErr := errors.New("transport unavailable")
	readErr := errors.New("response body failed")
	partial := agentTestSSE(t, agentTestChunk(agentTestTextDelta("partial")))
	tests := []struct {
		name       string
		reply      agentTestReply
		contains   string
		cause      error
		wantDeltas []string
	}{
		{
			name: "transport", reply: agentTestReply{err: transportErr},
			contains: "Error on inference start", cause: transportErr,
		},
		{
			name: "provider", reply: agentTestStream(t,
				agentTestChunk(agentTestTextDelta("partial")),
				components.ChatStreamChunk{Error: &components.ChatStreamChunkError{Code: 503, Message: "provider unavailable"}},
				agentTestChunk(agentTestTextDelta("must not be delivered")),
			),
			contains: "Error from provider: provider unavailable", wantDeltas: []string{"partial"},
		},
		{
			name: "malformed event", reply: agentTestBodyReply(strings.NewReader(partial + "data: {broken\n\ndata: [DONE]\n\n")),
			contains: "inference stream error", wantDeltas: []string{"partial"},
		},
		{
			name: "body read failure", reply: agentTestBodyReply(io.MultiReader(strings.NewReader(partial), agentTestErrorReader{readErr})),
			contains: "inference stream error", cause: readErr, wantDeltas: []string{"partial"},
		},
		{
			name: "truncated stream", reply: agentTestBodyReply(strings.NewReader(partial)),
			contains: "inference stream error", cause: io.ErrUnexpectedEOF, wantDeltas: []string{"partial"},
		},
		{
			name: "non-streaming response", reply: agentTestReply{
				contentType: "application/json",
				body:        &agentTestBody{Reader: strings.NewReader(`{"id":"reply","object":"chat.completion","created":0,"model":"test-model","choices":[],"system_fingerprint":null}`)},
			},
			contains: "Expected a streaming response",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			agent, _ := newAgentTestClient(t, tt.reply)
			var deltas []string
			message, err := agent.RunInference(t.Context(), []components.ChatMessages{agentTestUser("Hi")}, func(delta string) {
				deltas = append(deltas, delta)
			})
			if err == nil || !strings.Contains(err.Error(), tt.contains) {
				t.Fatalf("error = %v, want containing %q", err, tt.contains)
			}
			if tt.cause != nil && !errors.Is(err, tt.cause) {
				t.Errorf("error = %v, want wrapping %v", err, tt.cause)
			}
			if !reflect.DeepEqual(message, components.ChatAssistantMessage{}) {
				t.Errorf("message = %+v, want zero message on error", message)
			}
			if !reflect.DeepEqual(deltas, tt.wantDeltas) {
				t.Errorf("deltas = %q, want %q", deltas, tt.wantDeltas)
			}
		})
	}
}

func TestAgentRunInferenceCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	agent, _ := newAgentTestClient(t, agentTestStream(t,
		agentTestChunk(agentTestTextDelta("first")),
		agentTestChunk(agentTestTextDelta("second")),
	))
	var deltas []string
	_, err := agent.RunInference(ctx, []components.ChatMessages{agentTestUser("Hi")}, func(delta string) {
		deltas = append(deltas, delta)
		cancel()
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want context.Canceled", err)
	}
	if !reflect.DeepEqual(deltas, []string{"first"}) {
		t.Errorf("deltas = %q, want only the first fragment", deltas)
	}
}

func TestAgentRunTurnWithoutTools(t *testing.T) {
	agent, client := newAgentTestClient(t, agentTestStream(t, agentTestChunk(agentTestTextDelta("Final"))))
	conversation := []components.ChatMessages{
		agentTestUser("Earlier question"),
		components.CreateChatMessagesAssistant(agentTestAssistant("Earlier answer")),
		agentTestUser("Next question"),
	}
	var deltas []string
	got, err := agent.RunTurn(t.Context(), conversation, func(delta string) { deltas = append(deltas, delta) })
	if err != nil {
		t.Fatal(err)
	}
	want := append([]components.ChatMessages{}, conversation...)
	want = append(want, components.CreateChatMessagesAssistant(agentTestAssistant("Final")))
	assertAgentJSON(t, got, want)
	assertAgentJSON(t, client.requests[0].Messages, conversation)
	if !reflect.DeepEqual(deltas, []string{"Final"}) {
		t.Errorf("deltas = %q, want [Final]", deltas)
	}
}

func TestAgentRunTurnToolRounds(t *testing.T) {
	first := agentTestTextDelta("Searching")
	first.ToolCalls = []components.ChatStreamToolCall{
		agentTestCallDelta(0, "first", "lookup", `{"step":1}`),
		agentTestCallDelta(1, "second", "lookup", `{"step":2}`),
	}
	second := agentTestTextDelta("Again")
	second.ToolCalls = []components.ChatStreamToolCall{agentTestCallDelta(0, "third", "lookup", `{"step":3}`)}
	agent, client := newAgentTestClient(t,
		agentTestStream(t, agentTestChunk(first)),
		agentTestStream(t, agentTestChunk(second)),
		agentTestStream(t, agentTestChunk(agentTestTextDelta("Done"))),
	)
	ctx := context.WithValue(t.Context(), struct{}{}, "tool context")
	var arguments []string
	agent.AddTools([]Tool{agentTestTool("lookup", func(gotCtx context.Context, args json.RawMessage) (string, error) {
		if gotCtx != ctx {
			t.Error("handler did not receive the turn's context")
		}
		arguments = append(arguments, string(args))
		return fmt.Sprintf("result %d", len(arguments)), nil
	})})
	var deltas []string
	conversation, err := agent.RunTurn(ctx, []components.ChatMessages{agentTestUser("Search")}, func(delta string) {
		deltas = append(deltas, delta)
	})
	if err != nil {
		t.Fatal(err)
	}
	wantFirst := agentTestAssistant("Searching")
	wantFirst.ToolCalls = []components.ChatToolCall{
		agentTestCall("first", "lookup", `{"step":1}`),
		agentTestCall("second", "lookup", `{"step":2}`),
	}
	wantSecond := agentTestAssistant("Again")
	wantSecond.ToolCalls = []components.ChatToolCall{agentTestCall("third", "lookup", `{"step":3}`)}
	want := []components.ChatMessages{
		agentTestUser("Search"),
		components.CreateChatMessagesAssistant(wantFirst),
		agentTestToolResult("first", "result 1"),
		agentTestToolResult("second", "result 2"),
		components.CreateChatMessagesAssistant(wantSecond),
		agentTestToolResult("third", "result 3"),
		components.CreateChatMessagesAssistant(agentTestAssistant("Done")),
	}
	assertAgentJSON(t, conversation, want)
	for i, length := range []int{1, 4, 6} {
		assertAgentJSON(t, client.requests[i].Messages, want[:length])
	}
	if !reflect.DeepEqual(arguments, []string{`{"step":1}`, `{"step":2}`, `{"step":3}`}) {
		t.Errorf("handler arguments = %q, want calls in response order", arguments)
	}
	if !reflect.DeepEqual(deltas, []string{"Searching", "Again", "Done"}) {
		t.Errorf("deltas = %q, want fragments from every inference round", deltas)
	}
}

func TestAgentRunTurnToolErrors(t *testing.T) {
	tests := []struct {
		name       string
		register   bool
		nilHandler bool
		arguments  string
		wantCalls  int
		wantResult string
	}{
		{name: "unknown tool", arguments: "{}", wantResult: `Tool error: unknown tool "lookup"`},
		{name: "nil handler", register: true, nilHandler: true, arguments: "{}", wantResult: `Tool error: unknown tool "lookup"`},
		{name: "invalid JSON", register: true, arguments: `{"path":`, wantResult: "Tool error: arguments must be valid JSON"},
		{name: "handler error", register: true, arguments: `{"path":"."}`, wantCalls: 1, wantResult: "Tool error: lookup failed"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			agent, client := newAgentTestClient(t,
				agentTestStream(t, agentTestToolChunk(agentTestCallDelta(0, "call", "lookup", tt.arguments))),
				agentTestStream(t, agentTestChunk(agentTestTextDelta("Recovered"))),
			)
			calls := 0
			if tt.register {
				tool := agentTestTool("lookup", nil)
				if !tt.nilHandler {
					tool.Execute = func(_ context.Context, args json.RawMessage) (string, error) {
						calls++
						if string(args) != tt.arguments {
							t.Errorf("arguments = %q, want %q", args, tt.arguments)
						}
						return "discard this output", errors.New("lookup failed")
					}
				}
				agent.AddTools([]Tool{tool})
			}
			got, err := agent.RunTurn(t.Context(), []components.ChatMessages{agentTestUser("Search")}, nil)
			if err != nil {
				t.Fatalf("tool error should be returned to the model: %v", err)
			}
			assistant := components.ChatAssistantMessage{
				Role:      components.ChatAssistantMessageRoleAssistant,
				ToolCalls: []components.ChatToolCall{agentTestCall("call", "lookup", tt.arguments)},
			}
			want := []components.ChatMessages{
				agentTestUser("Search"),
				components.CreateChatMessagesAssistant(assistant),
				agentTestToolResult("call", tt.wantResult),
				components.CreateChatMessagesAssistant(agentTestAssistant("Recovered")),
			}
			assertAgentJSON(t, got, want)
			assertAgentJSON(t, client.requests[1].Messages, want[:3])
			if calls != tt.wantCalls {
				t.Errorf("handler calls = %d, want %d", calls, tt.wantCalls)
			}
		})
	}
}

func TestAgentRunTurnCancellation(t *testing.T) {
	for _, name := range []string{"before inference", "between tools", "after last tool"} {
		t.Run(name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			var replies []agentTestReply
			wantCalls := 0
			if name == "before inference" {
				cancel()
			} else {
				calls := []components.ChatStreamToolCall{agentTestCallDelta(0, "first", "lookup", "{}")}
				if name == "between tools" {
					calls = append(calls, agentTestCallDelta(1, "second", "lookup", "{}"))
				}
				replies = append(replies, agentTestStream(t, agentTestToolChunk(calls...)))
				wantCalls = 1
			}
			agent, _ := newAgentTestClient(t, replies...)
			calls := 0
			agent.AddTools([]Tool{agentTestTool("lookup", func(context.Context, json.RawMessage) (string, error) {
				calls++
				cancel()
				return "canceled", nil
			})})
			conversation, err := agent.RunTurn(ctx, []components.ChatMessages{agentTestUser("Search")}, nil)
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("error = %v, want context.Canceled", err)
			}
			if conversation != nil {
				t.Errorf("conversation = %+v, want nil on error", conversation)
			}
			if calls != wantCalls {
				t.Errorf("handler calls = %d, want %d", calls, wantCalls)
			}
		})
	}
}

func TestAgentRunTurnInferenceFailure(t *testing.T) {
	for _, afterTool := range []bool{false, true} {
		t.Run(fmt.Sprintf("after_tool=%t", afterTool), func(t *testing.T) {
			cause := errors.New("inference unavailable")
			var replies []agentTestReply
			if afterTool {
				replies = append(replies, agentTestStream(t, agentTestToolChunk(agentTestCallDelta(0, "call", "lookup", "{}"))))
			}
			replies = append(replies, agentTestReply{err: cause})
			agent, _ := newAgentTestClient(t, replies...)
			calls := 0
			agent.AddTools([]Tool{agentTestTool("lookup", func(context.Context, json.RawMessage) (string, error) {
				calls++
				return "result", nil
			})})
			conversation, err := agent.RunTurn(t.Context(), []components.ChatMessages{agentTestUser("Search")}, nil)
			if !errors.Is(err, cause) {
				t.Fatalf("error = %v, want wrapping %v", err, cause)
			}
			if conversation != nil {
				t.Errorf("conversation = %+v, want nil on error", conversation)
			}
			wantCalls := 0
			if afterTool {
				wantCalls = 1
			}
			if calls != wantCalls {
				t.Errorf("handler calls = %d, want %d", calls, wantCalls)
			}
		})
	}
}

func TestAgentAddToolsReplacesRegistry(t *testing.T) {
	for _, clear := range []bool{false, true} {
		t.Run(fmt.Sprintf("clear=%t", clear), func(t *testing.T) {
			calls := []components.ChatStreamToolCall{agentTestCallDelta(0, "old-call", "old", "{}")}
			if !clear {
				calls = append(calls, agentTestCallDelta(1, "new-call", "new", "{}"))
			}
			agent, client := newAgentTestClient(t,
				agentTestStream(t, agentTestToolChunk(calls...)),
				agentTestStream(t, agentTestChunk(agentTestTextDelta("Done"))),
			)
			agent.AddTools([]Tool{agentTestTool("old", func(context.Context, json.RawMessage) (string, error) {
				t.Error("replaced handler was executed")
				return "old result", nil
			})})
			var replacements []Tool
			newCalls := 0
			if !clear {
				replacements = []Tool{agentTestTool("new", func(context.Context, json.RawMessage) (string, error) {
					newCalls++
					return "new result", nil
				})}
			} else {
				replacements = []Tool{}
			}
			agent.AddTools(replacements)
			conversation, err := agent.RunTurn(t.Context(), []components.ChatMessages{agentTestUser("Search")}, nil)
			if err != nil {
				t.Fatal(err)
			}
			assertAgentJSON(t, conversation[2], agentTestToolResult("old-call", `Tool error: unknown tool "old"`))
			for _, request := range client.requests {
				if clear {
					if len(request.Tools) != 0 {
						t.Errorf("advertised tools = %+v, want none", request.Tools)
					}
				} else {
					assertAgentJSON(t, request.Tools, []components.ChatFunctionTool{replacements[0].Definition})
				}
			}
			if !clear {
				assertAgentJSON(t, conversation[3], agentTestToolResult("new-call", "new result"))
				if newCalls != 1 {
					t.Errorf("new handler calls = %d, want 1", newCalls)
				}
			}
		})
	}
}

func TestAgentRunEOF(t *testing.T) {
	for _, verbose := range []bool{false, true} {
		t.Run(fmt.Sprintf("verbose=%t", verbose), func(t *testing.T) {
			agent, _ := newAgentTestClient(t)
			agent.verbose = verbose
			agent.getUserInput = func() (string, bool) { return "", false }
			output, err := captureAgentOutput(t, func() error { return agent.Run(t.Context()) })
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(output, "Chatting with test-model") {
				t.Errorf("output = %q, want model banner", output)
			}
			if verbose && !strings.Contains(output, "Ending chat session") {
				t.Errorf("output = %q, want session completion log", output)
			}
		})
	}
}

func TestAgentRunPreservesHistory(t *testing.T) {
	agent, client := newAgentTestClient(t,
		agentTestStream(t, agentTestToolChunk(agentTestCallDelta(0, "call", "lookup", "{}"))),
		agentTestStream(t, agentTestChunk(agentTestTextDelta("好🙂"))),
		agentTestStream(t, agentTestChunk(agentTestTextDelta("OK"))),
	)
	agent.verbose = true
	input := []string{"Find files", "Thanks"}
	reads := 0
	agent.getUserInput = func() (string, bool) {
		reads++
		if reads > len(input) {
			return "", false
		}
		return input[reads-1], true
	}
	toolCalls := 0
	agent.AddTools([]Tool{agentTestTool("lookup", func(context.Context, json.RawMessage) (string, error) {
		toolCalls++
		return "files found", nil
	})})
	output, err := captureAgentOutput(t, func() error { return agent.Run(t.Context()) })
	if err != nil {
		t.Fatal(err)
	}
	wantCall := components.ChatAssistantMessage{
		Role:      components.ChatAssistantMessageRoleAssistant,
		ToolCalls: []components.ChatToolCall{agentTestCall("call", "lookup", "{}")},
	}
	want := []components.ChatMessages{
		agentTestUser("Find files"),
		components.CreateChatMessagesAssistant(wantCall),
		agentTestToolResult("call", "files found"),
		components.CreateChatMessagesAssistant(agentTestAssistant("好🙂")),
		agentTestUser("Thanks"),
	}
	for i, length := range []int{1, 3, 5} {
		assertAgentJSON(t, client.requests[i].Messages, want[:length])
	}
	if reads != 3 || toolCalls != 1 {
		t.Errorf("input reads = %d, tool calls = %d; want 3 and 1", reads, toolCalls)
	}
	for _, text := range []string{"Chatting with test-model", "好🙂", "OK", "Ending chat session"} {
		if !strings.Contains(output, text) {
			t.Errorf("output = %q, want containing %q", output, text)
		}
	}
}

func TestAgentRunInferenceFailureDrainsOutput(t *testing.T) {
	agent, _ := newAgentTestClient(t, agentTestStream(t,
		agentTestChunk(agentTestTextDelta("部分🙂")),
		components.ChatStreamChunk{Error: &components.ChatStreamChunkError{Code: 503, Message: "provider failed"}},
	))
	agent.verbose = true
	reads := 0
	agent.getUserInput = func() (string, bool) {
		reads++
		return "Hi", reads == 1
	}
	output, err := captureAgentOutput(t, func() error { return agent.Run(t.Context()) })
	if err == nil || !strings.Contains(err.Error(), "Error from provider: provider failed") {
		t.Fatalf("error = %v, want provider failure", err)
	}
	if !strings.Contains(output, "部分🙂\n") {
		t.Errorf("output = %q, want complete rendered text followed by a newline", output)
	}
	if reads != 1 {
		t.Errorf("input reads = %d, want 1 after inference failure", reads)
	}
}

// The fake replaces the SDK's HTTP boundary, so requests and SSE decoding still
// exercise the real SDK without network access or API credentials.
type agentTestHTTPClient struct {
	t        *testing.T
	replies  []agentTestReply
	requests []agentTestRequest
}

type agentTestRequest struct {
	Model    string                        `json:"model"`
	Stream   bool                          `json:"stream"`
	Messages []components.ChatMessages     `json:"messages"`
	Tools    []components.ChatFunctionTool `json:"tools"`
	ctx      context.Context
}

type agentTestReply struct {
	contentType string
	body        *agentTestBody
	err         error
}

type agentTestBody struct {
	io.Reader
	closed bool
}

func (b *agentTestBody) Close() error {
	b.closed = true
	return nil
}

type agentTestErrorReader struct{ err error }

func (r agentTestErrorReader) Read([]byte) (int, error) { return 0, r.err }

func (c *agentTestHTTPClient) Do(request *http.Request) (*http.Response, error) {
	c.t.Helper()
	defer request.Body.Close()
	if request.Method != http.MethodPost || request.URL.String() != "https://agent.test/chat/completions" {
		c.t.Fatalf("unexpected request: %s %s", request.Method, request.URL)
	}
	var decoded agentTestRequest
	if err := json.NewDecoder(request.Body).Decode(&decoded); err != nil {
		c.t.Fatalf("decode request: %v", err)
	}
	decoded.ctx = request.Context()
	index := len(c.requests)
	c.requests = append(c.requests, decoded)
	if index >= len(c.replies) {
		c.t.Fatalf("unexpected inference request %d", index+1)
	}
	reply := c.replies[index]
	if reply.err != nil {
		return nil, reply.err
	}
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": {reply.contentType}},
		Body:       reply.body,
		Request:    request,
	}, nil
}

func newAgentTestClient(t *testing.T, replies ...agentTestReply) (*Agent, *agentTestHTTPClient) {
	t.Helper()
	client := &agentTestHTTPClient{t: t, replies: replies}
	t.Cleanup(func() {
		if len(client.requests) != len(replies) {
			t.Errorf("inference requests = %d, want %d", len(client.requests), len(replies))
		}
		for i, reply := range replies {
			if i < len(client.requests) && reply.body != nil && !reply.body.closed {
				t.Errorf("response body %d was not closed", i+1)
			}
		}
	})
	sdk := openrouter.New(
		openrouter.WithClient(client),
		openrouter.WithServerURL("https://agent.test"),
		openrouter.WithSecurity("test-key"),
		openrouter.WithRetryConfig(retry.Config{Strategy: "none"}),
	)
	return NewAgent(*sdk, func() (string, bool) { return "", false }, "test-model", false), client
}

func agentTestBodyReply(reader io.Reader) agentTestReply {
	return agentTestReply{contentType: "text/event-stream", body: &agentTestBody{Reader: reader}}
}

func agentTestSSE(t *testing.T, chunks ...components.ChatStreamChunk) string {
	t.Helper()
	var body strings.Builder
	for _, chunk := range chunks {
		chunk.ID = "test-response"
		chunk.Model = "test-model"
		chunk.Object = components.ChatStreamChunkObjectChatCompletionChunk
		data, err := json.Marshal(chunk)
		if err != nil {
			t.Fatalf("marshal stream fixture: %v", err)
		}
		fmt.Fprintf(&body, "data: %s\n\n", data)
	}
	return body.String()
}

func agentTestStream(t *testing.T, chunks ...components.ChatStreamChunk) agentTestReply {
	t.Helper()
	return agentTestBodyReply(strings.NewReader(agentTestSSE(t, chunks...) + "data: [DONE]\n\n"))
}

func agentTestChunk(delta components.ChatStreamDelta) components.ChatStreamChunk {
	return components.ChatStreamChunk{Choices: []components.ChatStreamChoice{{Index: 0, Delta: delta}}}
}

func agentTestTextDelta(text string) components.ChatStreamDelta {
	delta := components.ChatStreamDelta{}
	delta.Content.Set(&text)
	return delta
}

func agentTestToolChunk(calls ...components.ChatStreamToolCall) components.ChatStreamChunk {
	return agentTestChunk(components.ChatStreamDelta{ToolCalls: calls})
}

func agentTestCallDelta(index int64, id, name, arguments string) components.ChatStreamToolCall {
	return components.ChatStreamToolCall{
		Index: index, ID: &id, Type: new(components.ChatStreamToolCallTypeFunction),
		Function: &components.ChatStreamToolCallFunction{Name: &name, Arguments: &arguments},
	}
}

func agentTestCall(id, name, arguments string) components.ChatToolCall {
	return components.ChatToolCall{
		ID: id, Type: components.ChatToolCallTypeFunction,
		Function: components.ChatToolCallFunction{Name: name, Arguments: arguments},
	}
}

func agentTestUser(text string) components.ChatMessages {
	return components.CreateChatMessagesUser(components.ChatUserMessage{
		Role: components.ChatUserMessageRoleUser, Content: components.CreateChatUserMessageContentStr(text),
	})
}

func agentTestAssistant(text string) components.ChatAssistantMessage {
	message := components.ChatAssistantMessage{Role: components.ChatAssistantMessageRoleAssistant}
	content := components.CreateChatAssistantMessageContentStr(text)
	message.Content.Set(&content)
	return message
}

func agentTestToolResult(id, text string) components.ChatMessages {
	return components.CreateChatMessagesTool(components.ChatToolMessage{
		Role: components.ChatToolMessageRoleTool, ToolCallID: id, Content: components.CreateChatToolMessageContentStr(text),
	})
}

func agentTestTool(name string, execute func(context.Context, json.RawMessage) (string, error)) Tool {
	return Tool{
		Definition: components.CreateChatFunctionToolChatFunctionToolFunction(components.ChatFunctionToolFunction{
			Type: components.ChatFunctionToolTypeFunction,
			Function: components.ChatFunctionToolFunctionFunction{
				Name: name, Description: new("Test tool"), Parameters: map[string]any{"type": "object"},
			},
		}),
		Execute: execute,
	}
}

func assertAgentJSON(t *testing.T, got, want any) {
	t.Helper()
	// Compare wire representations to avoid differences in SDK union bookkeeping
	// or nil versus empty optional fields after a request has been decoded.
	encode := func(value any) ([]byte, any) {
		t.Helper()
		data, err := json.Marshal(value)
		if err != nil {
			t.Fatalf("marshal comparison value: %v", err)
		}
		var decoded any
		if err := json.Unmarshal(data, &decoded); err != nil {
			t.Fatalf("decode comparison value: %v", err)
		}
		return data, decoded
	}
	gotJSON, gotValue := encode(got)
	wantJSON, wantValue := encode(want)
	if !reflect.DeepEqual(gotValue, wantValue) {
		t.Errorf("got  %s\nwant %s", gotJSON, wantJSON)
	}
}

func captureAgentOutput(t *testing.T, run func() error) (string, error) {
	t.Helper()
	// Session tests are deliberately serial because stdout and the logger are
	// process-wide. A file avoids pipe backpressure while the renderer runs.
	output, err := os.CreateTemp(t.TempDir(), "agent-output")
	if err != nil {
		t.Fatal(err)
	}
	stdout, logOutput := os.Stdout, log.Writer()
	os.Stdout = output
	log.SetOutput(output)
	defer func() {
		os.Stdout = stdout
		log.SetOutput(logOutput)
		output.Close()
	}()
	runErr := run()
	if _, err := output.Seek(0, io.SeekStart); err != nil {
		t.Fatal(err)
	}
	data, err := io.ReadAll(output)
	if err != nil {
		t.Fatal(err)
	}
	return string(data), runErr
}
