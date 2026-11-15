package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"time"

	openrouter "github.com/OpenRouterTeam/go-sdk"
	"github.com/OpenRouterTeam/go-sdk/models/components"
)

type Agent struct {
	client       openrouter.OpenRouter
	model        string
	getUserInput func() (string, bool)
	verbose      bool
	definitions  []components.ChatFunctionTool
	handlers     map[string]func(context.Context, json.RawMessage) (string, error)
}

func NewAgent(client openrouter.OpenRouter, getUserMessage func() (string, bool), model string, verbose bool) *Agent {
	return &Agent{
		client:       client,
		getUserInput: getUserMessage,
		model:        model,
		verbose:      verbose,
	}
}

func (a *Agent) AddTools(tools []Tool) {
	a.definitions = make([]components.ChatFunctionTool, 0, len(tools))
	a.handlers = make(map[string]func(context.Context, json.RawMessage) (string, error), len(tools))
	for _, tool := range tools {
		a.definitions = append(a.definitions, tool.Definition)
		a.handlers[tool.Definition.ChatFunctionToolFunction.Function.Name] = tool.Execute
	}
}

func (a *Agent) Run(ctx context.Context) error {
	conversation := []components.ChatMessages{}

	if a.verbose {
		log.Println("Starting chat session...")
	}

	fmt.Printf("Chatting with %s\n", a.model)

	for {
		fmt.Print("\u001b[94mYou\u001b[0m: ")

		userMessage, ok := a.getUserInput()

		if !ok {
			if a.verbose {
				log.Println("User input ended. Exiting loop")
			}

			break
		}

		if userMessage == "" {
			if a.verbose {
				log.Println("Skipping empty message")
			}
		}

		if a.verbose {
			log.Printf("User message received %q", userMessage)
		}

		chatUserMessage := []components.ChatMessages{
			components.CreateChatMessagesUser(
				components.ChatUserMessage{
					Content: components.CreateChatUserMessageContentStr(
						userMessage,
					),
					Role: components.ChatUserMessageRoleUser,
				},
			),
		}

		conversation = append(conversation, chatUserMessage...)

		if a.verbose {
			log.Println("Sending message to provider...")
		}

		chunks := make(chan string, 64)
		done := make(chan struct{})

		render := func(chunks <-chan string, done chan<- struct{}) {
			defer close(done)

			ticker := time.NewTicker(16 * time.Millisecond)
			defer ticker.Stop()

			frames := []string{"|", "/", "-", "\\"}
			frame := 0
			started := false

			spinnerTicker := time.NewTicker(64 * time.Millisecond)
			defer spinnerTicker.Stop()

			var pending []rune

			for chunks != nil || len(pending) > 0 {
				select {
				case <-spinnerTicker.C:
					if !started {
						fmt.Printf("\r%s Thinking...", frames[frame])
						frame = (frame + 1) % len(frames)
					}
				case delta, ok := <-chunks:
					if !ok {
						chunks = nil
					} else {
						pending = append(pending, []rune(delta)...)
					}
				case <-ticker.C:
					n := min(2, len(pending))
					if n > 0 {
						if !started {
							spinnerTicker.Stop()
							fmt.Print("\r\x1b[2K")
							started = true
						}
						fmt.Print(string(pending[:n]))
						pending = pending[n:]
					}
				}
			}
		}

		go render(chunks, done)

		onDelta := func(delta string) {
			chunks <- delta
		}

		nextConversation, err := a.RunTurn(ctx, conversation, onDelta)
		close(chunks)
		<-done
		fmt.Println()

		if err != nil {
			if a.verbose {
				log.Printf("Error during inference")
			}
			return err
		}

		conversation = nextConversation
	}

	if a.verbose {
		log.Println("Ending chat session")
	}

	return nil
}

func (a *Agent) RunInference(ctx context.Context, conversation []components.ChatMessages, onDelta func(string)) (components.ChatAssistantMessage, error) {
	res, err := a.client.Chat.Send(ctx, components.ChatRequest{
		Model:    &a.model,
		Messages: conversation,
		Stream:   new(true),
		Tools:    a.definitions,
	}, nil)

	if err != nil {
		return components.ChatAssistantMessage{}, fmt.Errorf("Error on inference start: %w", err)
	}

	if res == nil || res.EventStream == nil {
		return components.ChatAssistantMessage{}, fmt.Errorf("Expected a streaming response")
	}

	stream := res.EventStream
	defer stream.Close()

	var text strings.Builder
	var calls []components.ChatToolCall
	positions := make(map[int64]int)

	for stream.Next() {
		event := stream.Value()

		if event == nil {
			continue
		}

		chunk := event.Data

		if chunk.Error != nil {
			return components.ChatAssistantMessage{}, fmt.Errorf("Error from provider: %s", chunk.Error.Message)
		}

		for _, choice := range chunk.Choices {
			if choice.Index != 0 {
				continue
			}

			if delta, ok := choice.Delta.Content.Get(); ok && delta != nil {
				text.WriteString(*delta)

				if onDelta != nil {
					onDelta(*delta)
				}
			}

			for _, delta := range choice.Delta.ToolCalls {
				pos, exists := positions[delta.Index]

				if !exists {
					pos = len(calls)
					positions[delta.Index] = pos
					calls = append(calls, components.ChatToolCall{
						Type: components.ChatToolCallTypeFunction,
					})
				}

				call := &calls[pos]

				if delta.ID != nil {
					call.ID = *delta.ID
				}

				if delta.Function != nil {
					if delta.Function.Name != nil {
						call.Function.Name += *delta.Function.Name
					}

					if delta.Function.Arguments != nil {
						call.Function.Arguments += *delta.Function.Arguments
					}
				}
			}
		}
	}

	if err := stream.Err(); err != nil {
		return components.ChatAssistantMessage{}, fmt.Errorf("inference stream error: %w", err)
	}

	message := components.ChatAssistantMessage{
		Role:      components.ChatAssistantMessageRoleAssistant,
		ToolCalls: calls,
	}

	if text.Len() > 0 {
		content := components.CreateChatAssistantMessageContentStr(text.String())
		message.Content.Set(&content)
	}

	return message, nil
}

func (a *Agent) RunTurn(ctx context.Context, conversation []components.ChatMessages, onDelta func(string)) ([]components.ChatMessages, error) {
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		message, err := a.RunInference(ctx, conversation, onDelta)

		if err != nil {
			return nil, err
		}

		conversation = append(conversation, components.CreateChatMessagesAssistant(message))

		if len(message.ToolCalls) == 0 {
			return conversation, nil
		}

		for _, call := range message.ToolCalls {
			if err := ctx.Err(); err != nil {
				return nil, err
			}

			handler, exists := a.handlers[call.Function.Name]

			var result string

			switch {
			case !exists || handler == nil:
				result = fmt.Sprintf("Tool error: unknown tool %q", call.Function.Name)
			case !json.Valid([]byte(call.Function.Arguments)):
				result = "Tool error: arguments must be valid JSON"
			default:
				output, err := handler(ctx, json.RawMessage(call.Function.Arguments))

				if err != nil {
					result = fmt.Sprintf("Tool error: %v", err)
				} else {
					result = output
				}
			}

			toolMessage := components.CreateChatMessagesTool(
				components.ChatToolMessage{
					Role:       components.ChatToolMessageRoleTool,
					ToolCallID: call.ID,
					Content:    components.CreateChatToolMessageContentStr(result),
				},
			)

			conversation = append(conversation, toolMessage)
		}
	}
}
