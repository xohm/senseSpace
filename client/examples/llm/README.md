# LLM Examples

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Examples](#examples)
  - [llm.py - OpenAI API](#llmpy---openai-api)
  - [llm_ollama.py - Local Ollama LLM](#llm_ollamapy---local-ollama-llm)
  - [llm_ollama_expert.py - Expert Pose Analysis](#llm_ollama_expertpy---expert-pose-analysis)

## Overview
These examples use a Large Language Model for retrieval of additional information about detected poses.

## Installation

### Python Dependencies
Install required packages:
```bash
pip install -r requirements.txt
```

## Examples

### llm.py - OpenAI API
Uses the OpenAI API to inquire pose analysis data.

**Setup:**
1. Get an API key from https://platform.openai.com/api-keys
2. Create a `.env` file in this directory:
   ```env
   OPENAI_API_KEY=sk-your-key-here
   OPENAI_MODEL=gpt-4o-mini
   ```
3. Run:
   ```bash
   python llm.py --server localhost --viz
   ```

**Keyboard shortcuts:**
- `SPACE` - Describe the current pose
- `A` - Analyze pose emotions/mood
- `B` - Suggest exercise or activity

---

### llm_ollama.py - Local Ollama LLM
Uses a local Ollama LLM for better latency, privacy, and no internet dependency or license payments.

**Setup:**

1. **Install Ollama:**
   
   **Linux:**
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```
   
   **macOS:**
   ```bash
   brew install ollama
   ```
   
   **Windows:**
   
   Download installer from https://ollama.com/download

2. **Pull a model** (choose one):
   ```bash
   ollama pull llama3.2        # Recommended: Fast & good quality (3B params)
   ollama pull llama3.1        # Better quality, slower (8B params)
   ollama pull qwen2.5:1.5b    # Smaller, faster (1.5B params)
   ```

3. **Configure** (optional - defaults work fine):
   
   Create/edit `.env` file:
   ```env
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama3.2
   ```

4. **Run:**
   ```bash
   python llm_ollama.py --server localhost --viz
   ```
   
   The script will automatically start Ollama if it's not running.

**Manual server start** (optional):
```bash
# Start Ollama server in separate terminal
ollama serve
```

**Keyboard shortcuts:**
- `SPACE` - Describe the current pose
- `A` - Analyze pose emotions/mood
- `B` - Suggest exercise or activity

---

## Comparison

| Feature | OpenAI (llm.py) | Ollama (llm_ollama.py) |
|---------|----------------|------------------------|
| **Cost** | ~$0.15/1M tokens | Free (local) |
| **Speed** | ~1-2s | ~2-5s (depends on hardware) |
| **Privacy** | Sends data to cloud | 100% local |
| **Internet** | Required | Not required |
| **Quality** | Excellent | Very good |
| **Setup** | API key needed | Model download (~2-4GB) |

## Troubleshooting

### Ollama
- **"Cannot connect to Ollama"**: Make sure Ollama is installed and run `ollama serve`
- **"Model not found"**: Run `ollama pull llama3.2`
- **Slow responses**: Use a smaller model like `qwen2.5:1.5b` or upgrade your hardware
- **Check Ollama status**: `ollama list` shows installed models

### OpenAI
- **"API key not found"**: Make sure `.env` file exists with valid `OPENAI_API_KEY`
- **Rate limits**: Upgrade your OpenAI plan or add delays between requests



