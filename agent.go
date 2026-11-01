package main

import (
	"context"
	"fmt"
	"log"

	openrouter "github.com/OpenRouterTeam/go-sdk"
	"github.com/OpenRouterTeam/go-sdk/models/components"
	"github.com/OpenRouterTeam/go-sdk/models/operations"
)

type Agent struct {
	client       openrouter.OpenRouter
	model        string
	getUserInput func() (string, bool)
	verbose      bool
}

func NewAgent(client openrouter.OpenRouter, getUserMessage func() (string, bool), model string, verbose bool) *Agent {
	return &Agent{
		client:       client,
		getUserInput: getUserMessage,
		model:        model,
		verbose:      verbose,
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

		res, err := a.RunInference(ctx, conversation)

		if err != nil {
			if a.verbose {
				log.Printf("Error during inference")
			}
		}

		if len(res.ChatResult.Choices) == 0 {
			continue
		}

		message := res.ChatResult.Choices[0].GetMessage()

		conversation = append(conversation, components.CreateChatMessagesAssistant(message))

		content, ok := message.Content.Get()

		if ok && content.Type == components.ChatAssistantMessageContentTypeStr && content.Str != nil {
			fmt.Println(*content.Str)
		}
	}

	if a.verbose {
		log.Println("Ending chat session")
	}

	return nil
}

func (a *Agent) RunInference(ctx context.Context, conversation []components.ChatMessages) (*operations.SendChatCompletionRequestResponse, error) {
	res, err := a.client.Chat.Send(ctx, components.ChatRequest{
		Model:    &a.model,
		Messages: conversation,
	}, nil)

	if a.verbose {
		if err != nil {
			log.Printf("Request failed with error %v", err)
		} else {
			log.Printf("Request completed successfully")
		}
	}

	return res, err
}
