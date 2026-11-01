package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"os"

	openrouter "github.com/OpenRouterTeam/go-sdk"
)

func main() {
	ctx := context.Background()

	verbose := flag.Bool("verbose", false, "enable verbose logging")

	if *verbose {
		log.SetOutput(os.Stderr)
		log.SetFlags(log.LstdFlags | log.Lshortfile)
		log.Println("Verbose logging enabled")
	} else {
		log.SetOutput(os.Stdout)
		log.SetFlags(0)
		log.SetPrefix("")
	}

	client := openrouter.New(
		openrouter.WithSecurity(os.Getenv("OPENROUTER_API_KEY")),
	)

	modelName := "deepseek/deepseek-v4.1-flash-1017"

	scanner := bufio.NewScanner(os.Stdin)
	getUserInput := func() (string, bool) {
		if !scanner.Scan() {
			return "", false
		}

		return scanner.Text(), true
	}

	agent := NewAgent(*client, getUserInput, modelName, *verbose)

	err := agent.Run(ctx)

	if err != nil {
		fmt.Printf("Error %s\n", err)
	}
}
