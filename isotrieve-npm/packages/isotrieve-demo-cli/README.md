# Isotrieve Zero-Friction Demo

Try the Agent Embedding Communication Protocol (Isotrieve) in 30 seconds. No API keys required.

## Features

- **Local Inference**: Uses local HuggingFace models via `@xenova/transformers`.
- **Zero Configuration**: No OpenAI/Anthropic keys needed.
- **Visual Comparison**: Side-by-side metrics of Text Handoff vs Vector Handoff.

## Usage

You can run this demo directly from the NPM registry (once published) or locally.

### 1. Direct Run (npx)
```bash
npx @isotrieve/demo-cli basic
```

### 2. Local Development
```bash
# From monorepo root
npm install
npm run build --workspace=@isotrieve/demo-cli

# Run the demo
node packages/isotrieve-demo-cli/dist/index.js basic
```

## Scenarios

- **basic**: Compares latency and privacy of text vs vector handoff.
- **rag**: (Coming soon) Simulates a Retrieve-Then-Read pipeline.
- **privacy**: (Coming soon) Demonstrates privacy boundary violation in text handoff.
