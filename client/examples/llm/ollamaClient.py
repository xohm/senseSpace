# -----------------------------------------------------------------------------
# Sense Space
# -----------------------------------------------------------------------------
# Ollama LLM Client
# -----------------------------------------------------------------------------
# IAD, Zurich University of the Arts / zhdk.ch
# Max Rheiner
# -----------------------------------------------------------------------------

import requests
import subprocess
import time
import platform
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Callable


class OllamaClient:
    """Simple Ollama client for LLM interactions"""
    
    def __init__(self, model_name="llama3.2:1b", ollama_url="http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.ready = False
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.platform = platform.system()  # 'Linux', 'Darwin' (macOS), 'Windows'
    
    def connect(self):
        """Initialize connection to Ollama server"""
        # Check if server is running
        if not self.check_server():
            print("[Ollama] Server not running, attempting to start...")
            if self.start_server():
                time.sleep(2)
            else:
                print("[Ollama] Failed to start server")
                return False
        
        # Verify connection
        if not self.check_server():
            print("[Ollama] Could not connect to server")
            return False
        
        print(f"[Ollama] Connected at {self.ollama_url}")
        
        # Check model availability
        if not self.check_model():
            print(f"[Ollama] Model '{self.model_name}' not available")
            print(f"[Ollama] Run: ollama pull {self.model_name}")
            return False
        
        print(f"[Ollama] Model loaded: {self.model_name}")
        self.ready = True
        return True
    
    def check_server(self):
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def start_server(self):
        """Start Ollama server"""
        try:
            import os
            if self.platform == "Windows":
                # On Windows, use CREATE_NO_WINDOW flag
                subprocess.Popen(
                    ['ollama', 'serve'],
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:
                # On Unix-like systems
                devnull = open(os.devnull, 'w')
                subprocess.Popen(
                    ['ollama', 'serve'],
                    stdout=devnull,
                    stderr=devnull,
                    close_fds=True
                )
            return True
        except:
            return False
    
    def check_model(self):
        """Check if model is available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                available = [m['name'] for m in models]
                return any(self.model_name in m for m in available)
            return False
        except:
            return False
    
    def generate(self, prompt, temperature=0.7, max_tokens=500, images=None):
        """
        Generate response from Ollama (synchronous)
        
        Args:
            prompt: User prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            images: List of base64-encoded images (optional)
            
        Returns:
            tuple: (response_string, error_string) - one will be None
        """
        if not self.ready:
            return None, "Client not ready"
        
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            # Add images if provided (for vision models)
            if images:
                payload["images"] = images
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', ''), None
            else:
                # Parse error details
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', response.text)
                except:
                    error_msg = response.text
                
                full_error = f"HTTP {response.status_code}: {error_msg}"
                print(f"[Ollama] {full_error}")
                
                # Add helpful suggestions for common errors
                if response.status_code == 500:
                    if "resource limitations" in error_msg.lower():
                        full_error += "\n\n💡 This model may require more RAM/VRAM. Try:"
                        full_error += "\n  • Use a smaller model (e.g., llama3.2:1b)"
                        full_error += "\n  • Check 'ollama ps' to see memory usage"
                        full_error += "\n  • Restart Ollama: 'ollama serve'"
                    elif "internal error" in error_msg.lower():
                        full_error += "\n\n💡 Check Ollama server logs:"
                        if self.platform == "Linux":
                            full_error += "\n  • journalctl -u ollama -f"
                        full_error += "\n  • Or restart: 'ollama serve'"
                
                return None, full_error
                
        except requests.exceptions.Timeout:
            error = "Request timeout (>60s)"
            print(f"[Ollama] {error}")
            return None, error
        except Exception as e:
            error = f"Request failed: {str(e)}"
            print(f"[Ollama] {error}")
            return None, error
    
    def generate_async(self, 
                      prompt: str, 
                      on_response: Callable[[str], None],
                      on_error: Optional[Callable[[str], None]] = None,
                      temperature: float = 0.7,
                      max_tokens: int = 500,
                      images: Optional[list] = None):
        """
        Generate response from Ollama (asynchronous)
        
        Args:
            prompt: User prompt
            on_response: Callback function called with response (str)
            on_error: Optional callback function called with error message (str)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            images: List of base64-encoded images (optional)
            
        Returns:
            Future object that can be used to cancel or check status
        """
        def _generate():
            response, error = self.generate(prompt, temperature, max_tokens, images)
            if response is not None:
                on_response(response)
            else:
                if on_error:
                    on_error(error or "Unknown error")
        
        return self.executor.submit(_generate)
    
    def list_models(self):
        """List available models"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [m['name'] for m in models]
            return []
        except:
            return []
    
    def shutdown(self):
        """Shutdown executor"""
        self.executor.shutdown(wait=False)
    
    def get_model_info(self, model_name=None):
        """Get detailed model information"""
        target = model_name or self.model_name
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                for m in models:
                    if target in m['name']:
                        size_gb = m.get('size', 0) / (1024**3)
                        return {
                            'name': m['name'],
                            'size': f"{size_gb:.1f}GB",
                            'size_bytes': m.get('size', 0),
                            'modified': m.get('modified_at', 'unknown')
                        }
            return None
        except:
            return None
    
    def list_models_with_info(self):
        """List available models with size information"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                result = []
                for m in models:
                    size_gb = m.get('size', 0) / (1024**3)
                    result.append({
                        'name': m['name'],
                        'size': f"{size_gb:.1f}GB",
                        'size_bytes': m.get('size', 0)
                    })
                return result
            return []
        except:
            return []
    
    def get_gpu_info(self):
        """Get GPU memory info if available (cross-platform)"""
        # Try NVIDIA first (Linux, Windows, macOS with eGPU)
        nvidia_info = self._get_nvidia_gpu_info()
        if nvidia_info:
            return nvidia_info
        
        # Try AMD (Linux, Windows)
        amd_info = self._get_amd_gpu_info()
        if amd_info:
            return amd_info
        
        # Try Apple Silicon (macOS)
        apple_info = self._get_apple_gpu_info()
        if apple_info:
            return apple_info
        
        return None
    
    def _get_nvidia_gpu_info(self):
        """Get NVIDIA GPU info via nvidia-smi"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                used, total = map(int, result.stdout.strip().split(','))
                return {
                    'type': 'NVIDIA',
                    'used_mb': used,
                    'total_mb': total,
                    'free_mb': total - used,
                    'used_pct': (used / total) * 100
                }
        except:
            pass
        return None
    
    def _get_amd_gpu_info(self):
        """Get AMD GPU info (Linux only, via rocm-smi)"""
        if self.platform != "Linux":
            return None
        try:
            result = subprocess.run(
                ['rocm-smi', '--showmeminfo', 'vram'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                # Parse rocm-smi output (format varies, this is a simplified version)
                # You might need to adjust based on actual output format
                return {
                    'type': 'AMD',
                    'used_mb': 0,  # Would need parsing
                    'total_mb': 0,  # Would need parsing
                    'free_mb': 0,
                    'used_pct': 0
                }
        except:
            pass
        return None
    
    def _get_apple_gpu_info(self):
        """Get Apple Silicon GPU info (macOS only)"""
        if self.platform != "Darwin":
            return None
        try:
            # Apple Silicon doesn't have discrete GPU memory reporting like NVIDIA
            # Unified memory architecture means CPU and GPU share RAM
            # We can only indicate that Apple Silicon is available
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0 and 'Apple' in result.stdout:
                return {
                    'type': 'Apple Silicon',
                    'used_mb': 0,  # Not separately trackable
                    'total_mb': 0,  # Shared with system RAM
                    'free_mb': 0,
                    'used_pct': 0,
                    'note': 'Unified memory (shared with CPU)'
                }
        except:
            pass
        return None
    
    def pull_model(self, model_name: str, on_progress: Optional[Callable[[str], None]] = None):
        """
        Download/pull an Ollama model (synchronous)
        
        Args:
            model_name: Name of model to pull (e.g., 'llama3.2:1b')
            on_progress: Optional callback for progress updates
            
        Returns:
            tuple: (success: bool, error_message: str or None)
        """
        try:
            if on_progress:
                on_progress(f"Starting download of {model_name}...")
            
            # Use UTF-8 encoding and handle errors gracefully for Windows
            process = subprocess.Popen(
                ['ollama', 'pull', model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',  # Replace problematic characters instead of failing
                bufsize=1
            )
            
            # Stream output in real-time
            for line in process.stdout:
                # Clean ANSI escape codes and special characters
                clean_line = self._clean_progress_line(line.rstrip())
                if clean_line:  # Only send non-empty lines
                    if on_progress:
                        on_progress(clean_line)
                    else:
                        print(clean_line, flush=True)
            
            process.wait()
            
            if process.returncode == 0:
                if on_progress:
                    on_progress(f"✅ Model '{model_name}' downloaded successfully")
                return True, None
            else:
                error = f"Failed to download model (exit code: {process.returncode})"
                if on_progress:
                    on_progress(f"❌ {error}")
                return False, error
                
        except FileNotFoundError:
            error = "Ollama not found. Please install Ollama first."
            if on_progress:
                on_progress(f"❌ {error}")
            return False, error
        except Exception as e:
            error = f"Download failed: {str(e)}"
            if on_progress:
                on_progress(f"❌ {error}")
            return False, error
    
    def _clean_progress_line(self, line: str) -> str:
        """
        Clean progress line from ANSI codes and special characters
        
        Args:
            line: Raw progress line from ollama
            
        Returns:
            Cleaned line suitable for display
        """
        # Remove ANSI escape codes (colors, cursor movements, etc.)
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        line = ansi_escape.sub('', line)
        
        # Remove carriage returns and other control characters
        line = line.replace('\r', '').replace('\x08', '')
        
        # Clean up multiple spaces
        line = re.sub(r'\s+', ' ', line).strip()
        
        return line
    
    def pull_model_async(self,
                        model_name: str,
                        on_progress: Optional[Callable[[str], None]] = None,
                        on_complete: Optional[Callable[[bool, Optional[str]], None]] = None):
        """
        Download/pull an Ollama model (asynchronous)
        
        Args:
            model_name: Name of model to pull (e.g., 'llama3.2:1b')
            on_progress: Optional callback for progress updates (str)
            on_complete: Optional callback when done (success: bool, error: str or None)
            
        Returns:
            Future object that can be used to cancel or check status
        """
        def _pull():
            success, error = self.pull_model(model_name, on_progress)
            if on_complete:
                on_complete(success, error)
        
        return self.executor.submit(_pull)
    
    def delete_model(self, model_name: str):
        """
        Delete/remove an Ollama model
        
        Args:
            model_name: Name of model to delete
            
        Returns:
            tuple: (success: bool, error_message: str or None)
        """
        try:
            result = subprocess.run(
                ['ollama', 'rm', model_name],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=10
            )
            
            if result.returncode == 0:
                return True, None
            else:
                error = result.stderr or result.stdout or "Unknown error"
                return False, error
                
        except FileNotFoundError:
            return False, "Ollama not found"
        except Exception as e:
            return False, str(e)
