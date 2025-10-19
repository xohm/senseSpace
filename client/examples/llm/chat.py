# -----------------------------------------------------------------------------
# Sense Space
# -----------------------------------------------------------------------------
# Ollama LLM Chat Example (Qt GUI)
# -----------------------------------------------------------------------------
# IAD, Zurich University of the Arts / zhdk.ch
# Max Rheiner
# -----------------------------------------------------------------------------

import sys
import base64
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QTextEdit, QLineEdit, QPushButton, QHBoxLayout,
                             QComboBox, QLabel, QFileDialog)
from PyQt5.QtCore import pyqtSignal, QObject, QTimer
from PyQt5.QtGui import QTextCursor, QFont, QPixmap

from ollamaClient import OllamaClient


class ChatSignals(QObject):
    """Signals for async communication with Qt"""
    response_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    models_loaded = pyqtSignal(list)


class ChatWindow(QMainWindow):
    """Simple Ollama chat window"""
    
    def __init__(self, model_name="llama3.2:1b"):
        super().__init__()
        self.client = OllamaClient(model_name=model_name)
        self.signals = ChatSignals()
        
        # Connect signals
        self.signals.response_received.connect(self.on_response)
        self.signals.error_occurred.connect(self.on_error)
        self.signals.models_loaded.connect(self.on_models_loaded)
        
        # Thinking animation
        self.thinking_timer = QTimer()
        self.thinking_timer.timeout.connect(self.update_thinking_animation)
        self.thinking_dots = 0
        self.thinking_message_pos = None
        
        # Attached image
        self.attached_image_path = None
        self.attached_image_base64 = None
        
        self.init_ui()
        self.init_ollama()
    
    def init_ui(self):
        """Setup UI components"""
        self.setWindowTitle("Ollama Chat")
        self.setGeometry(100, 100, 800, 600)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Model selector at top
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        
        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.model_combo)
        
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_models)
        model_layout.addWidget(self.refresh_button)
        
        self.info_button = QPushButton("Info")
        self.info_button.clicked.connect(self.show_model_info)
        model_layout.addWidget(self.info_button)
        
        self.load_button = QPushButton("Load Model")
        self.load_button.clicked.connect(self.load_new_model)
        model_layout.addWidget(self.load_button)
        
        model_layout.addStretch()
        layout.addLayout(model_layout)
        
        # Chat display (read-only) - NOW SUPPORTS MARKDOWN
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Courier", 10))
        self.chat_display.setAcceptRichText(True)  # Enable rich text
        layout.addWidget(self.chat_display)
        
        # Attachment display
        self.attachment_label = QLabel()
        self.attachment_label.setMaximumHeight(100)
        self.attachment_label.setScaledContents(True)
        self.attachment_label.hide()
        layout.addWidget(self.attachment_label)
        
        # Input area
        input_layout = QHBoxLayout()
        
        self.attach_button = QPushButton("📎 Attach")
        self.attach_button.clicked.connect(self.attach_file)
        input_layout.addWidget(self.attach_button)
        
        self.clear_attach_button = QPushButton("✖")
        self.clear_attach_button.clicked.connect(self.clear_attachment)
        self.clear_attach_button.hide()
        input_layout.addWidget(self.clear_attach_button)
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your message and press Enter...")
        self.input_field.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.input_field)
        
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)
        
        layout.addLayout(input_layout)
        
        # Status display
        self.append_system("Initializing Ollama connection...")
    
    def init_ollama(self):
        """Initialize Ollama connection"""
        if self.client.connect():
            self.append_system("✅ Connected to Ollama")
            self.refresh_models()
            self.append_system("Ready! Type your message below.\n")
        else:
            self.append_system("❌ Failed to connect to Ollama")
    
    def refresh_models(self):
        """Refresh available models list"""
        models = self.client.list_models()
        if models:
            self.signals.models_loaded.emit(models)
        else:
            self.append_system("⚠️ No models found. Run: ollama pull <model-name>")
    
    def on_models_loaded(self, models):
        """Populate model dropdown"""
        current = self.model_combo.currentText()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(models)
        
        # Try to restore previous selection or use client's current model
        if current and current in models:
            self.model_combo.setCurrentText(current)
        elif self.client.model_name in models:
            self.model_combo.setCurrentText(self.client.model_name)
        
        self.model_combo.blockSignals(False)
        self.append_system(f"📋 Found {len(models)} models")
    
    def on_model_changed(self, model_name):
        """Handle model selection change"""
        if not model_name or model_name == self.client.model_name:
            return
        
        self.append_system(f"🔄 Switching to model: {model_name}")
        self.client.model_name = model_name
        self.client.ready = True
        self.setWindowTitle(f"Ollama Chat - {model_name}")
        self.append_system(f"✅ Now using: {model_name}\n")
    
    def show_model_info(self):
        """Show detailed information about current model and system"""
        self.append_system("📊 System Information")
        self.append_system("=" * 50)
        
        # Current model info
        model_info = self.client.get_model_info()
        if model_info:
            self.append_system(f"Current Model: {model_info['name']}")
            self.append_system(f"Size: {model_info['size']}")
            self.append_system(f"Modified: {model_info['modified']}")
        else:
            self.append_system(f"Current Model: {self.client.model_name}")
        
        # GPU info
        gpu_info = self.client.get_gpu_info()
        if gpu_info:
            self.append_system(f"\nGPU: {gpu_info['type']}")
            if gpu_info.get('note'):
                self.append_system(f"Note: {gpu_info['note']}")
            else:
                self.append_system(f"Memory: {gpu_info['used_mb']}MB / {gpu_info['total_mb']}MB ({gpu_info['used_pct']:.1f}% used)")
                self.append_system(f"Free: {gpu_info['free_mb']}MB")
        else:
            self.append_system("\nGPU: Not detected (using CPU)")
        
        # Available models
        models_info = self.client.list_models_with_info()
        if models_info:
            self.append_system(f"\nAvailable Models ({len(models_info)}):")
            for info in models_info:
                marker = "→" if info['name'] == self.client.model_name else " "
                self.append_system(f"  {marker} {info['name']} ({info['size']})")
        
        self.append_system("=" * 50 + "\n")
    
    def load_new_model(self):
        """Load a new model from Ollama registry"""
        from PyQt5.QtWidgets import QInputDialog
        
        model_name, ok = QInputDialog.getText(
            self,
            "Load Model",
            "Enter model name to download:\n(e.g., llama3.2:1b, phi3.5:mini, qwen2.5:1.5b)",
            QLineEdit.Normal,
            ""
        )
        
        if ok and model_name.strip():
            model_name = model_name.strip()
            self.append_system(f"📥 Downloading model: {model_name}")
            self.append_system("This may take a while depending on model size...\n")
            
            # Track progress line position
            self.chat_display.append("[PROGRESS] Starting download...")
            self.download_progress_pos = self.chat_display.textCursor().position()
            self.chat_display.moveCursor(QTextCursor.End)
            
            # Disable UI during download
            self.load_button.setEnabled(False)
            self.input_field.setEnabled(False)
            self.send_button.setEnabled(False)
            
            # Start download in background thread
            def download_task():
                try:
                    import subprocess
                    
                    process = subprocess.Popen(
                        ['ollama', 'pull', model_name],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1
                    )
                    
                    output_lines = []
                    for line in process.stdout:
                        stripped = line.strip()
                        output_lines.append(stripped)
                        # Send progress updates via signal
                        if stripped:
                            self.signals.error_occurred.emit(f"PROGRESS:{stripped}")
                    
                    process.wait()
                    return process.returncode == 0, '\n'.join(output_lines), ""
                    
                except Exception as e:
                    return False, "", str(e)
            
            def on_download_complete(result):
                # Remove progress line
                if hasattr(self, 'download_progress_pos'):
                    cursor = self.chat_display.textCursor()
                    cursor.setPosition(self.download_progress_pos)
                    cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
                    cursor.removeSelectedText()
                    delattr(self, 'download_progress_pos')
                
                success, stdout, stderr = result
                if success:
                    self.append_system(f"✅ Successfully downloaded: {model_name}")
                    self.append_system("Refreshing model list...")
                    self.refresh_models()
                else:
                    self.append_error(f"Failed to download model:\n{stderr if stderr else stdout}")
                
                # Re-enable UI
                self.load_button.setEnabled(True)
                self.input_field.setEnabled(True)
                self.send_button.setEnabled(True)
            
            # Run download in executor
            future = self.client.executor.submit(download_task)
            
            # Monitor completion (using a timer to check)
            def check_download():
                if future.done():
                    timer.stop()
                    try:
                        result = future.result()
                        on_download_complete(result)
                    except Exception as e:
                        self.append_error(f"Download error: {e}")
                        self.load_button.setEnabled(True)
                        self.input_field.setEnabled(True)
                        self.send_button.setEnabled(True)
            
            timer = QTimer()
            timer.timeout.connect(check_download)
            timer.start(1000)  # Check every second
    
    def append_system(self, message):
        """Append system message to chat"""
        self.chat_display.append(f"[SYSTEM] {message}")
        self.chat_display.moveCursor(QTextCursor.End)
    
    def append_user(self, message):
        """Append user message to chat"""
        self.chat_display.append(f"\n[YOU] {message}")
        self.chat_display.moveCursor(QTextCursor.End)
    
    def markdown_to_html(self, markdown_text):
        """Convert markdown to HTML for display"""
        try:
            import markdown
            return markdown.markdown(
                markdown_text,
                extensions=['fenced_code', 'codehilite', 'tables']
            )
        except ImportError:
            # Fallback: basic formatting if markdown package not available
            return markdown_text.replace('\n', '<br>').replace('**', '<b>').replace('*', '<i>')
    
    def append_assistant(self, message):
        """Append assistant message to chat (with Markdown rendering)"""
        html = self.markdown_to_html(message)
        self.chat_display.append(f"\n<b>[ASSISTANT]</b><br>{html}<br>")
        self.chat_display.moveCursor(QTextCursor.End)
    
    def append_error(self, message):
        """Append error message to chat"""
        self.chat_display.append(f"\n❌ [ERROR] {message}\n")
        self.chat_display.moveCursor(QTextCursor.End)
    
    def start_thinking_animation(self):
        """Start the 'thinking' animation"""
        self.thinking_dots = 0
        # Add initial thinking message
        self.chat_display.append(f"\n[ASSISTANT] Thinking")
        self.thinking_message_pos = self.chat_display.textCursor().position()
        self.chat_display.moveCursor(QTextCursor.End)
        # Start timer (update every 500ms)
        self.thinking_timer.start(500)
    
    def update_thinking_animation(self):
        """Update the thinking animation dots"""
        self.thinking_dots = (self.thinking_dots + 1) % 4
        dots = "." * self.thinking_dots
        
        # Move to thinking message and update it
        cursor = self.chat_display.textCursor()
        cursor.setPosition(self.thinking_message_pos)
        cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        cursor.removeSelectedText()
        
        # Re-add the message with animated dots
        self.chat_display.insertPlainText(f"Thinking{dots}")
        self.chat_display.moveCursor(QTextCursor.End)
    
    def stop_thinking_animation(self):
        """Stop the thinking animation"""
        self.thinking_timer.stop()
        
        # Remove the thinking message
        if self.thinking_message_pos:
            cursor = self.chat_display.textCursor()
            cursor.setPosition(self.thinking_message_pos)
            cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
            cursor.removeSelectedText()
            self.thinking_message_pos = None
    
    def attach_file(self):
        """Open file dialog to attach an image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.gif *.bmp);;All Files (*)"
        )
        
        if file_path:
            try:
                # Read and encode image
                with open(file_path, 'rb') as f:
                    image_data = f.read()
                    self.attached_image_base64 = base64.b64encode(image_data).decode('utf-8')
                
                self.attached_image_path = file_path
                
                # Show thumbnail with preserved aspect ratio
                pixmap = QPixmap(file_path)
                if not pixmap.isNull():
                    from PyQt5.QtCore import Qt
                    scaled_pixmap = pixmap.scaled(
                        self.attachment_label.width(), 
                        100, 
                        Qt.KeepAspectRatio, 
                        Qt.SmoothTransformation
                    )
                    self.attachment_label.setPixmap(scaled_pixmap)
                    self.attachment_label.setScaledContents(False)  # Don't stretch
                    self.attachment_label.show()
                    self.clear_attach_button.show()
                    self.append_system(f"📎 Attached: {Path(file_path).name}")
                else:
                    self.append_error(f"Could not load image: {file_path}")
                    
            except Exception as e:
                self.append_error(f"Error attaching file: {e}")
    
    def clear_attachment(self):
        """Clear attached image"""
        self.attached_image_path = None
        self.attached_image_base64 = None
        self.attachment_label.clear()
        self.attachment_label.hide()
        self.clear_attach_button.hide()
        self.append_system("📎 Attachment cleared")
    
    def send_message(self):
        """Send user message to Ollama"""
        if not self.client.ready:
            self.append_error("Ollama not ready")
            return
        
        message = self.input_field.text().strip()
        if not message and not self.attached_image_base64:
            return
        
        # Default message if only image
        if not message:
            message = "What's in this image?"
        
        # Clear input
        self.input_field.clear()
        
        # Display user message
        if self.attached_image_path:
            self.append_user(f"{message} [📷 {Path(self.attached_image_path).name}]")
        else:
            self.append_user(message)
        
        # Start thinking animation
        self.start_thinking_animation()
        
        # Disable input while processing
        self.input_field.setEnabled(False)
        self.send_button.setEnabled(False)
        self.attach_button.setEnabled(False)
        self.model_combo.setEnabled(False)
        
        # Prepare images list
        images = [self.attached_image_base64] if self.attached_image_base64 else None
        
        # Clear attachment after sending
        self.clear_attachment()
        
        # Call async generate with callbacks
        self.client.generate_async(
            prompt=message,
            images=images,
            on_response=lambda resp: self.signals.response_received.emit(resp),
            on_error=lambda err: self.signals.error_occurred.emit(err)
        )
    
    def on_response(self, response):
        """Handle LLM response"""
        # Stop thinking animation
        self.stop_thinking_animation()
        
        self.append_assistant(response)
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
        self.attach_button.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.input_field.setFocus()
    
    def on_error(self, error):
        """Handle LLM error"""
        # Check if this is a progress update (hack using error signal)
        if error.startswith("PROGRESS:"):
            progress_line = error[9:]  # Remove "PROGRESS:" prefix
            
            # Update the progress line in place
            if hasattr(self, 'download_progress_pos'):
                cursor = self.chat_display.textCursor()
                cursor.setPosition(self.download_progress_pos)
                cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
                cursor.removeSelectedText()
                self.chat_display.insertPlainText(f"[PROGRESS] {progress_line}")
                self.chat_display.moveCursor(QTextCursor.End)
            return
        
        # Stop thinking animation
        self.stop_thinking_animation()
        
        self.append_error(error)
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
        self.attach_button.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.input_field.setFocus()
    
    def closeEvent(self, event):
        """Cleanup on close"""
        self.thinking_timer.stop()
        self.client.shutdown()
        event.accept()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ollama Chat GUI")
    parser.add_argument("--model", default="llama3.2:1b", 
                       help="Ollama model name (default: llama3.2:1b)")
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    window = ChatWindow(model_name=args.model)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

