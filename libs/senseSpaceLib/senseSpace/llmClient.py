# -----------------------------------------------------------------------------
# Sense Space
# -----------------------------------------------------------------------------
# Ollama LLM Wrapper Class (Generic Expert System)
# -----------------------------------------------------------------------------
# IAD, Zurich University of the Arts / zhdk.ch
# Max Rheiner
# -----------------------------------------------------------------------------
#
# This class provides a general-purpose wrapper around the local Ollama server.
# It supports:
#   - Expert systems defined by JSON (context, task rules, examples)
#   - Persistent context through Ollama's /api/chat endpoint (faster inference)
#   - Synchronous and asynchronous calls (with callback support)
#   - Direct model access (no expert context)
#   - Automatic model download and Ollama startup
#
# -----------------------------------------------------------------------------

import time
import json
import requests
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Optional


class LLMClient:
    """Generic wrapper for running expert or general LLM systems through Ollama."""

    def __init__(self, model_name=None, expert_json=None, auto_download=False, verbose=False):
        self.ollama_url = "http://localhost:11434"
        self.ollama_ready = False
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.expert_messages = []
        self.ollama_options = {}
        self.input_format = "detailed"
        self.auto_download = auto_download
        self.verbose = verbose  # Store verbose flag
        
        # Load expert config first (may override model_name)
        if expert_json:
            self.load_expert_config(expert_json)
        
        # Only use command-line model if expert config didn't set one
        if model_name is not None:
            self.model_name = model_name
            if self.verbose:
                print(f"[LOAD] Model overridden: {self.model_name}")
        elif not hasattr(self, 'model_name'):
            # No expert config and no command-line arg
            self.model_name = "llama3.2:1b"

    # --------------------------------------------------------------------------
    # Initialization / connection
    # --------------------------------------------------------------------------

    def on_init(self):
        """Initialize Ollama connection and ensure model availability."""
        print("[INIT] Checking Ollama server...")
        if not self._check_ollama_server():
            print("[INIT] Ollama server not running, attempting to start...")
            if not self._start_ollama_server():
                print("[ERROR] Could not start Ollama server.")
                return
            time.sleep(2)

        if self._check_ollama_server():
            print(f"[INIT] Ollama running at {self.ollama_url}")
            if self._check_model_available(auto_download=self.auto_download):
                self.ollama_ready = True
                print(f"[INIT] Using model: {self.model_name}")
            else:
                print(f"[ERROR] Model '{self.model_name}' unavailable.")
                print(f"[INFO] Run manually: ollama pull {self.model_name}")
        else:
            print("[ERROR] Could not connect to Ollama server.")

    def _check_ollama_server(self):
        try:
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def _start_ollama_server(self):
        try:
            subprocess.Popen(['ollama', 'serve'],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            return True
        except Exception as e:
            print(f"[ERROR] Failed to start Ollama: {e}")
            return False

    def _check_model_available(self, auto_download=True):
        """Check if model exists locally, otherwise download it."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = [m['name'] for m in response.json().get('models', [])]
                available = any(self.model_name in m for m in models)
                if available:
                    print(f"[INIT] Model '{self.model_name}' found locally.")
                    return True
                elif auto_download:
                    print(f"[INIT] Model '{self.model_name}' not found. Downloading...")
                    if self._download_model():
                        print(f"[INIT] Successfully downloaded '{self.model_name}'.")
                        return True
                    else:
                        print(f"[ERROR] Download failed for '{self.model_name}'.")
                        return False
                else:
                    print(f"[WARN] Model '{self.model_name}' missing (auto_download=False).")
                    return False
            return False
        except Exception as e:
            print(f"[ERROR] Could not query models: {e}")
            return False

    def _download_model(self):
        """Download model using 'ollama pull'"""
        try:
            subprocess.run(['ollama', 'pull', self.model_name],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.STDOUT,
                           check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    # --------------------------------------------------------------------------
    # Expert configuration
    # --------------------------------------------------------------------------

    def load_expert(self, source):
        """Load expert definition from JSON file or dict."""
        try:
            if isinstance(source, str):
                print(f"[LOAD] Loading expert config from {source}")
                with open(source, "r", encoding="utf-8") as f:
                    self.expert = json.load(f)
            elif isinstance(source, dict):
                self.expert = source
            else:
                raise ValueError("Expert source must be JSON path or dict")

            # Build persistent system context
            ctx = self.expert.get("context", {})
            sys_text = (
                self.expert.get("description", "") + "\n\n" +
                "Environment:\n" + ctx.get("environment", "") + "\n" +
                "Coordinate system:\n" + ctx.get("coordinate_system", "") + "\n" +
                "Rules:\n" + "\n".join(ctx.get("task_rules", []))
            )

            self.messages = [{'role': 'system', 'content': sys_text}]
            self.params = self.expert.get("parameters", {})

            # Add few-shot examples (optional)
            examples = self.expert.get("response", {}).get("examples", [])
            for ex in examples:
                self.messages.append({'role': 'user', 'content': ex['input']})
                self.messages.append({'role': 'assistant', 'content': ex['output']})

            self._chat_history = None  # reset previous chat context
            print(f"[LOAD] Expert context loaded: {self.expert.get('name','Unknown')}")
            print(f"[LOAD] Model: {self.expert.get('model', self.model_name)}")

        except Exception as e:
            print(f"[ERROR] Failed to load expert definition: {e}")

    def load_expert_config(self, json_path: str):
        """Load expert system configuration from JSON"""
        try:
            with open(json_path, 'r') as f:
                config = json.load(f)
            
            if self.verbose:
                print(f"[LOAD] Loading expert config from {json_path}")
            
            # Override model name if specified in config
            if "model" in config:
                self.model_name = config["model"]
                if self.verbose:
                    print(f"[LOAD] Model from config: {self.model_name}")
            
            # Build system message from config
            system_content = self._build_system_message(config)
            
            self.expert_messages = [
                {"role": "system", "content": system_content}
            ]
            
            # Store ollama options if provided
            self.ollama_options = config.get("ollama_options", {})
            
            if self.verbose:
                print(f"[LOAD] Expert: {config.get('name', 'Unknown')}")
                if self.ollama_options:
                    print(f"[LOAD] Options: num_predict={self.ollama_options.get('num_predict', 'default')}, "
                          f"ctx={self.ollama_options.get('num_ctx', 'default')}")
            
        except FileNotFoundError:
            print(f"[ERROR] Config not found: {json_path}")
            self.expert_messages = []
            self.ollama_options = {}
        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON: {e}")
            self.expert_messages = []
            self.ollama_options = {}

    def reset_context(self):
        """Reset conversation context to initial system message"""
        if self.expert_messages and len(self.expert_messages) > 0:
            # Keep only the system message (first message)
            system_msg = self.expert_messages[0]
            self.expert_messages = [system_msg]
            print("[CHAT] Conversation context reset")
        else:
            print("[WARN] No expert context to reset")

    # --------------------------------------------------------------------------
    # Optimized chat-based request (persistent context)
    # --------------------------------------------------------------------------

    def _llm_chat_request(self, user_input):
        """Use Ollama's /api/chat for persistent context and faster responses."""
        if not hasattr(self, "_chat_history") or self._chat_history is None:
            self._chat_history = self.messages.copy()

        # Append new user message
        self._chat_history.append({"role": "user", "content": user_input})

        payload = {
            "model": self.model_name,
            "messages": self._chat_history,
            "stream": False,
            "options": self.params
        }

        try:
            t0 = time.perf_counter()
            r = requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=15)
            if r.status_code == 200:
                data = r.json()
                answer = data.get("message", {}).get("content", "").strip()
                latency = (time.perf_counter() - t0) * 1000
                print(f"[CHAT] {latency:.0f} ms | {answer}")

                # Append assistant reply to persistent chat history
                self._chat_history.append({"role": "assistant", "content": answer})
                return answer
            else:
                print(f"[ERROR] Ollama chat returned {r.status_code}: {r.text}")
                return ""
        except Exception as e:
            print(f"[ERROR] Chat request failed: {e}")
            return ""

    def _call_ollama_chat(self, messages: list) -> Optional[str]:
        """Internal: Call Ollama chat API"""
        try:
            default_options = {
                "temperature": 0.3,
                "top_p": 0.9,
                "num_predict": 100,
                "stop": ["\n\n\n"]
            }
            
            options = {**default_options, **self.ollama_options}
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": options
            }
            
            # Suppress Ollama verbose output
            import sys
            from io import StringIO
            
            # Save original stderr/stdout
            old_stderr = sys.stderr
            old_stdout = sys.stdout
            
            # Redirect to null if not in verbose mode
            if not self.verbose:
                sys.stderr = StringIO()
                sys.stdout = StringIO()
            
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.ollama_url}/api/chat",
                    json=payload,
                    timeout=30
                )
                elapsed_ms = int((time.time() - start_time) * 1000)
            finally:
                # Always restore stderr/stdout
                sys.stderr = old_stderr
                sys.stdout = old_stdout
            
            if response.status_code == 200:
                result = response.json()
                assistant_message = result.get("message", {}).get("content", "").strip()
                
                if not assistant_message:
                    if self.verbose:
                        print(f"[WARN] Empty response")
                    return None
                
                # Remove [END] marker if present
                assistant_message = assistant_message.replace("[END]", "").strip()
                
                # Only show timing if verbose OR if there's actually output
                if self.verbose or assistant_message:
                    print(f"[{elapsed_ms}ms] reply time")
                
                return assistant_message
            else:
                error_msg = response.json().get("error", "Unknown error")
                print(f"[ERROR] {response.status_code}: {error_msg}")
                return None
                
        except requests.exceptions.Timeout:
            print("[ERROR] Timeout")
            return None
        except Exception as e:
            print(f"[ERROR] {e}")
            return None

    # --------------------------------------------------------------------------
    # Expert Calls
    # --------------------------------------------------------------------------

    def call_expert_sync(self, input_data: str):
        """Perform a synchronous expert call (with persistent context)."""
        if not self.ollama_ready or not self.expert:
            print("[ERROR] LLM not ready or expert not loaded.")
            return ""
        return self._llm_chat_request(input_data)

    def call_expert_async(self, user_input: str, callback=None):
        """
        Call the expert system asynchronously with user input
        Stateless - only sends system message + current input
        """
        if not self.ollama_ready:
            print("[ERROR] Ollama not ready.")
            if callback:
                callback(None)
            return
        
        # Build user message
        user_message = {
            "role": "user",
            "content": user_input
        }
        
        # Always stateless: system message + current input only
        if len(self.expert_messages) > 0:
            system_msg = self.expert_messages[0]
            messages_to_send = [system_msg, user_message]
        else:
            messages_to_send = [user_message]
        
        # Submit to thread pool
        future = self.executor.submit(
            self._call_ollama_chat,
            messages_to_send
        )
        
        if callback:
            future.add_done_callback(lambda f: callback(f.result()))

    # --------------------------------------------------------------------------
    # Generic Model Calls (no context)
    # --------------------------------------------------------------------------

    def call_model_sync(self, prompt: str):
        """Direct model call without expert context (blocking)."""
        if not self.ollama_ready:
            print("[ERROR] Ollama not ready.")
            return ""

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": self.params or {
                "temperature": 0.3,
                "num_predict": 200,
                "top_p": 0.9
            }
        }

        try:
            t0 = time.perf_counter()
            r = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=30)
            if r.status_code == 200:
                result = r.json().get("response", "").strip()
                latency = (time.perf_counter() - t0) * 1000
                print(f"[LLM] {latency:.0f} ms | {result}")
                return result
            else:
                print(f"[ERROR] Ollama returned {r.status_code}: {r.text}")
                return ""
        except Exception as e:
            print(f"[ERROR] Model request failed: {e}")
            return ""

    def call_model_async(self, prompt: str, callback=None):
        """Async general call without context."""
        if not self.ollama_ready:
            print("[ERROR] Ollama not ready.")
            return None

        def worker():
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": self.params or {
                    "temperature": 0.3,
                    "num_predict": 200,
                    "top_p": 0.9
                }
            }
            try:
                r = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=30)
                if r.status_code == 200:
                    result = r.json().get("response", "").strip()
                    if callback:
                        try:
                            callback(result)
                        except Exception as cb_err:
                            print(f"[WARN] Callback error: {cb_err}")
                    return result
                else:
                    print(f"[ERROR] Ollama returned {r.status_code}: {r.text}")
                    return ""
            except Exception as e:
                print(f"[ERROR] Model request failed: {e}")
                return ""

        return self.executor.submit(worker)

    # --------------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------------

    def shutdown(self):
        """Close thread pool cleanly."""
        self.executor.shutdown(wait=False)
        print("[LLM] Thread pool shut down.")

    def _build_system_message(self, config: dict) -> str:
        """Build system message from expert configuration"""
        parts = []
        
        # Add role and task clearly
        if "context" in config:
            context = config["context"]
            if "role" in context:
                parts.append(context['role'])
            if "task" in context:
                parts.append(context['task'])
        
        # Add rules as bullet points
        if "rules" in config:
            parts.append("\nRules:")
            for rule in config["rules"]:
                parts.append(f"• {rule}")
        
        # Add examples WITHOUT the User:/You: format (just show input → output)
        if "examples" in config and len(config["examples"]) > 0:
            parts.append("\nExamples:")
            for example in config["examples"]:
                parts.append(f"Input: {example['input']}")
                parts.append(f"Output: {example['output']}")
                parts.append("")  # blank line between examples
        
        return "\n".join(parts)
