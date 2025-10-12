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


class LLMClient:
    """Generic wrapper for running expert or general LLM systems through Ollama."""

    def __init__(self, model_name="phi4-mini", expert_json=None, auto_download=True):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.ollama_url = "http://localhost:11434"
        self.model_name = model_name
        self.auto_download = auto_download
        self.ollama_ready = False

        # Expert context
        self.expert = None
        self.messages = []
        self.params = {}
        self._chat_history = None  # persistent context for performance

        if expert_json:
            self.load_expert(expert_json)
        else:
            print("[INIT] No expert definition provided yet.")

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

    def reset_context(self):
        """Clear the persistent chat memory."""
        self._chat_history = None
        print("[CHAT] Context cleared.")

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

    # --------------------------------------------------------------------------
    # Expert Calls
    # --------------------------------------------------------------------------

    def call_expert_sync(self, input_data: str):
        """Perform a synchronous expert call (with persistent context)."""
        if not self.ollama_ready or not self.expert:
            print("[ERROR] LLM not ready or expert not loaded.")
            return ""
        return self._llm_chat_request(input_data)

    def call_expert_async(self, input_data: str, callback=None):
        """Asynchronous expert call using persistent chat context."""
        if not self.ollama_ready or not self.expert:
            print("[ERROR] LLM not ready or expert not loaded.")
            return None

        def worker():
            res = self._llm_chat_request(input_data)
            if callback:
                try:
                    callback(res)
                except Exception as cb_err:
                    print(f"[WARN] Callback error: {cb_err}")
            return res

        return self.executor.submit(worker)

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
