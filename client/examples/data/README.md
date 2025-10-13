# Expert Configuration

This document describes the structure and purpose of the expert configuration JSON files used by the SenseSpace LLM system.

## Overview

Expert configurations define how an LLM analyzes sensor data (e.g., pose tracking) in a specific context. They provide:

- **System context** - What the LLM needs to know about the environment and data
- **Task definition** - What the LLM should do with the data
- **Response format** - How the LLM should structure its output
- **Model settings** - Which model to use and how to configure it
- **Optimization parameters** - Runtime settings for the client application

## Configuration Structure

### Core Fields

| Key                    | Type   | Required | Description                                                                                                                          |
| ---------------------- | ------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **`name`**             | string | Yes      | A unique, human-readable name for the expert configuration (used in logs or UI).                                                     |
| **`description`**      | string | No       | A short description of what this expert model does and in which environment it operates.                                             |
| **`model`**            | string | No       | Name of the Ollama model to use (e.g., `llama3.2:1b`, `phi4-mini`). Can be overridden by command-line arguments.                    |

### Context Section

The `context` section provides the LLM with persistent knowledge about its role and environment.

| Key                    | Type   | Required | Description                                                                                                                          |
| ---------------------- | ------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **`context`**          | object | Yes      | Contains the expert's *persistent knowledge* — setup, coordinate system, and task constraints.                                       |
|   `role`               | string | Yes      | Defines what the LLM is (e.g., "You are a motion analysis assistant").                                                               |
|   `task`               | string | Yes      | Defines what the LLM should do (e.g., "Describe the body pose in ONE sentence").                                                     |
|   `environment`        | string | No       | Textual description of the system setup (e.g., camera layout, sensors, or input sources).                                            |
|   `coordinate_system`  | string | No       | Defines axis orientation, units, and angular conventions.                                                                            |
|   `data_schema`        | object | No       | Lists and describes all expected numeric input fields. These names correspond to the pose data keys sent to the LLM.                 |

### Rules Section

The `rules` array defines behavioral constraints for the LLM's responses.

| Key                    | Type   | Required | Description                                                                                                                          |
| ---------------------- | ------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **`rules`**            | array  | No       | List of strings describing how the LLM should behave (e.g., "Write only ONE sentence", "Maximum 20 words").                          |

**Example:**
```json
"rules": [
  "Start with the body position (standing, sitting, crouching)",
  "Mention the most notable arm, leg, or head positions",
  "ONE sentence only, maximum 20 words",
  "Use natural language"
]
```

### Response Format Section

The `response_format` object (legacy) or direct fields define output structure. Modern configs typically use `rules` instead.

| Key                    | Type   | Required | Description                                                                                                                          |
| ---------------------- | ------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **`response_format`**  | object | No       | Legacy: Defines how the model should phrase and structure its output.                                                                |
|   `style`              | string | No       | High-level style guide for phrasing (e.g., "concise", "factual", "ultra-brief").                                                     |
|   `length`             | string | No       | Desired response length (e.g., "1 sentence, max 10 words").                                                                          |
|   `tone`               | string | No       | Desired tone (e.g., "professional", "casual").                                                                                       |

### Examples Section

The `examples` array provides few-shot learning examples to teach the LLM the expected behavior.

| Key                    | Type   | Required | Description                                                                                                                          |
| ---------------------- | ------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **`examples`**         | array  | No       | Few-shot learning examples — pairs of input pose data and expected output text. Helps the model learn tone and behavior.             |
|   `input`              | string | Yes      | Example input data (should match the format sent to the LLM).                                                                        |
|   `output`             | string | Yes      | Expected response for the given input.                                                                                               |

**Example:**
```json
"examples": [
  {
    "input": "Subject in STANDING position. Head level, facing forward.",
    "output": "Standing upright with head facing forward."
  },
  {
    "input": "Subject in CROUCHING position. Left arm extended horizontally.",
    "output": "Crouching with left arm extended outward."
  }
]
```

### Ollama Options Section

The `ollama_options` object contains parameters passed directly to the Ollama API for model generation control.

| Key                    | Type   | Required | Description                                                                                                                          |
| ---------------------- | ------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **`ollama_options`**   | object | No       | Model generation settings passed directly to Ollama.                                                                                 |
|   `temperature`        | float  | No       | Controls randomness (0.0 = deterministic, 1.0 = very creative). Default: 0.3. Range: 0.0-2.0.                                       |
|   `top_p`              | float  | No       | Nucleus sampling cutoff (0.0-1.0). Lower = safer, more focused output. Default: 0.9.                                                 |
|   `top_k`              | int    | No       | Limits token choices to top K candidates. Lower = more focused. Default: 40.                                                         |
|   `num_predict`        | int    | No       | Maximum number of tokens (words) to generate per response. Default: 100.                                                             |
|   `num_ctx`            | int    | No       | Context window size (number of tokens kept in memory). Larger = better understanding but slower. Default: 2048.                      |
|   `num_batch`          | int    | No       | Batch size for processing. Default: 128.                                                                                             |
|   `num_gpu`            | int    | No       | Number of GPU layers to use (99 = all layers on GPU). Default: 99.                                                                   |
|   `stop`               | array  | No       | List of strings that stop generation when encountered (e.g., `["\n\n", "Input:"]`).                                                  |

**Tuning Guide:**
- **For speed**: Lower `num_predict` (25-50), smaller `num_ctx` (512-1024), higher `temperature` (0.3-0.5)
- **For quality**: Higher `num_predict` (100-200), larger `num_ctx` (4096-8192), lower `temperature` (0.1-0.3)
- **For reliability**: Remove aggressive stop tokens, use `temperature` 0.3-0.5, `top_p` 0.8-0.9

### Input Format Section

The `input_format` field (optional) controls how pose data is formatted before sending to the LLM.

| Key                    | Type   | Required | Description                                                                                                                          |
| ---------------------- | ------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **`input_format`**     | string | No       | How to format input data. Options: `"brief"` (semantic only), `"detailed"` (all data). Default: `"detailed"`.                       |

**Note:** Modern implementations send full skeletal data regardless of this setting for best accuracy.

### Optimization Section (Legacy)

The `optimization` object contains runtime tuning for real-time interaction. **These are not sent to the LLM** but used by the Python client.

| Key                    | Type   | Required | Description                                                                                                                          |
| ---------------------- | ------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **`optimization`**     | object | No       | Runtime tuning for real-time interaction loops. Not sent to LLM.                                                                     |
|   `stream`             | bool   | No       | If true, responses are streamed token-by-token for lower latency perception. Not currently implemented.                              |
|   `max_history`        | int    | No       | Number of conversation turns to keep in memory. Default: 0 (stateless for pose analysis).                                           |
|   `update_interval_ms` | int    | No       | How frequently new data frames are sent (in milliseconds). Controlled by user interaction, not config.                               |
|   `target_latency_ms`  | int    | No       | Target expected response latency for monitoring. Not enforced, informational only.                                                   |

## Example Configuration

```json
{
  "name": "PoseAnalysisExpert",
  "description": "Reliable pose analyzer with full skeletal data",
  "model": "llama3.2:1b",
  
  "context": {
    "role": "You are a motion analysis assistant.",
    "task": "Based on the skeletal data provided, write ONE clear sentence describing what the person is doing."
  },
  
  "rules": [
    "Start with the body position (standing, sitting, crouching)",
    "Mention the most notable arm, leg, or head positions",
    "ONE sentence only, maximum 20 words",
    "Be specific and descriptive",
    "Use natural language"
  ],
  
  "examples": [
    {
      "input": "Subject detected (confidence: 85.2%)\n\nDetected pose features:\n• Body position: STANDING\n• Head facing forward\n• Left arm extended horizontally",
      "output": "Standing upright with left arm extended horizontally."
    }
  ],
  
  "ollama_options": {
    "temperature": 0.5,
    "top_p": 0.9,
    "num_predict": 50,
    "num_ctx": 4096,
    "stop": ["\n\n"]
  }
}
```

## Usage

### Command Line

```bash
# Use default config
python llm_ollama_expert.py --server 192.168.1.5 --viz

# Use custom config
python llm_ollama_expert.py --server 192.168.1.5 --viz --expert path/to/config.json

# Override model from command line
python llm_ollama_expert.py --server 192.168.1.5 --viz --model llama3.2:3b
```

### Python API

```python
from senseSpaceLib.senseSpace.llmClient import LLMClient

# Load expert config (includes model selection)
llm_client = LLMClient(
    expert_json="data/expert_pose_config.json",
    auto_download=True
)

# Or override model
llm_client = LLMClient(
    model_name="llama3.2:3b",
    expert_json="data/expert_pose_config.json"
)
```

## Tuning for Your Use Case

### Fast, Real-time Analysis
- Model: `tinyllama` or `llama3.2:1b`
- `num_predict`: 25-35
- `num_ctx`: 512-1024
- `temperature`: 0.2-0.4
- Expected latency: 100-300ms

### Balanced (Recommended)
- Model: `llama3.2:1b`
- `num_predict`: 50
- `num_ctx`: 4096
- `temperature`: 0.5
- Expected latency: 500-1000ms

### High Quality, Detailed Analysis
- Model: `llama3.2:3b` or `phi4-mini`
- `num_predict`: 100-150
- `num_ctx`: 8192
- `temperature`: 0.3-0.5
- Expected latency: 2-5 seconds

## Troubleshooting

### Empty Responses
- Check stop tokens - remove aggressive ones like `"."` or `"\n"`
- Increase `num_predict` to allow longer responses
- Try higher `temperature` (0.4-0.6)

### Model Echoes Input
- Remove confusing examples or simplify them
- Add stop tokens: `["Input:", "Output:", "Subject detected"]`
- Use a better model (llama3.2:1b instead of tinyllama)

### Slow Performance
- Reduce `num_ctx` (1024 or 2048)
- Reduce `num_predict` (25-50)
- Use smaller model (llama3.2:1b instead of 3b)
- Set `num_gpu: 99` to ensure GPU usage

### VRAM Issues
- Use smaller model (llama3.2:1b ~1.3GB, tinyllama ~0.6GB)
- Reduce `num_ctx` to 512 or 1024
- Close other GPU applications
