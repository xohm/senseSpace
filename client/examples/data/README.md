# Expert Predefinition

| Key                    | Type   | Description                                                                                                                          |
| ---------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| **`name`**             | string | A unique, human-readable name for the expert configuration (used in logs or UI).                                                     |
| **`description`**      | string | A short description of what this expert model does and in which environment it operates.                                             |
| **`context`**          | object | Contains the expert’s *persistent knowledge* — setup, coordinate system, and task constraints.                                       |
|   `environment`        | string | Textual description of the system setup (e.g., camera layout, sensors, or input sources).                                            |
|   `coordinate_system`  | string | Defines axis orientation, units, and angular conventions.                                                                            |
|   `data_schema`        | object | Lists and describes all expected numeric input fields. These names correspond to the pose data keys sent to the LLM.                 |
|   `task_rules`         | array  | Behavioral rules the LLM should follow when analyzing poses (e.g., brevity, certainty, phrasing).                                    |
| **`response`**         | object | Defines how the model should phrase and structure its output.                                                                        |
|   `style`              | string | High-level style guide for phrasing (e.g., lowercase, concise, factual).                                                             |
|   `examples`           | array  | Few-shot learning examples — pairs of input pose data and expected short text outputs. These help the model learn tone and behavior. |
| **`model`**            | string | Name of the Ollama model to use (e.g., `phi4-mini`, `phi4`, `llama3.2`).                                                             |
| **`parameters`**       | object | Model generation settings passed directly to Ollama.                                                                                 |
|   `temperature`        | float  | Controls randomness (lower = more deterministic).                                                                                    |
|   `num_predict`        | int    | Maximum number of tokens (words) to generate per response.                                                                           |
|   `top_p`              | float  | Nucleus sampling cutoff; lower = safer, more focused output.                                                                         |
|   `num_ctx`            | int    | Context window size (number of tokens kept in memory).                                                                               |
|   `stop`               | array  | List of token stop conditions (commonly `["\\n"]`).                                                                                  |
| **`optimization`**     | object | Runtime tuning for real-time interaction loops. These are *not sent to the LLM* but used by your Python client.                      |
|   `stream`             | bool   | If true, responses are streamed token-by-token for lower latency.                                                                    |
|   `max_history`        | int    | Number of conversation turns to keep in memory (helps maintain continuity).                                                          |
|   `update_interval_ms` | int    | How frequently new data frames are sent (in milliseconds).                                                                           |
|   `target_latency_ms`  | int    | Target expected response latency (for monitoring or throttling).                                                                     |
