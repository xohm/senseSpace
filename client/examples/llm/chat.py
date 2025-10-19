# -----------------------------------------------------------------------------
# Sense Space
# -----------------------------------------------------------------------------
# Ollama LLM Chat Example (Qt GUI)
# -----------------------------------------------------------------------------
# IAD, Zurich University of the Arts / zhdk.ch
# Max Rheiner
# -----------------------------------------------------------------------------
#
# Features:
# - Multi-model support with dropdown selection
# - Vision models (e.g., moondream) for image analysis
# - Text models (e.g., llama3.2, phi4) for conversation
# - Image and text file attachments
# - Conversation history with context sharing between models
# - Real-time model downloading with progress indicator
# - GPU/CPU system information display
# - Markdown rendering for formatted responses
# - Thinking animation during inference
#
# Usage:
#   python chat.py                    # Start with default model (llama3.2:1b)
#   python chat.py --model phi4       # Start with specific model
#
# Controls:
#   📎 Attach    - Attach images or text files
#   💬 Share Context - Toggle conversation history sharing between models
#   Info         - Display system and model information
#   Load Model   - Download new models from Ollama registry
#   Refresh      - Refresh available models list
#
# Tips:
#   - Use moondream with images, then switch to llama for questions
#   - Keep "Share Context" enabled to maintain conversation across models
#   - Attachments are auto-cleared when switching models
#   - Text files are included in the prompt, images sent as base64
#
# Requirements:
#   - Ollama server running (ollama serve)
#   - PyQt5, requests, markdown
# -----------------------------------------------------------------------------

import sys
import base64
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QTextEdit, QLineEdit, QPushButton, QHBoxLayout,
                             QComboBox, QLabel, QFileDialog, QCheckBox)
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
        
        # Attached files
        self.attached_image_path = None
        self.attached_image_base64 = None
        self.attached_text_content = None
        self.attached_file_type = None  # 'image' or 'text'
        
        # Conversation history
        self.conversation_history = []
        self.use_context = True  # Default: enabled
        
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
        
        # Context checkbox
        self.context_checkbox = QCheckBox("💬 Share Context")
        self.context_checkbox.setChecked(True)
        self.context_checkbox.setToolTip("Send conversation history to new models")
        self.context_checkbox.toggled.connect(self.on_context_toggled)
        model_layout.addWidget(self.context_checkbox)
        
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
        
        # Replace QLineEdit with QTextEdit for multi-line input
        self.input_field = QTextEdit()
        self.input_field.setPlaceholderText("Type your message (Enter to send, Shift+Enter for new line)...")
        self.input_field.setMaximumHeight(100)
        self.input_field.setAcceptRichText(False)
        self.input_field.installEventFilter(self)  # Install event filter for key handling
        input_layout.addWidget(self.input_field)
        
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)
        
        layout.addLayout(input_layout)
        
        # Status display
        self.append_system("Initializing Ollama connection...")
    
    def eventFilter(self, obj, event):
        """Handle key events for input field"""
        if obj == self.input_field and event.type() == event.KeyPress:
            from PyQt5.QtCore import Qt
            
            # Enter without Shift -> send message
            if event.key() == Qt.Key_Return and not (event.modifiers() & Qt.ShiftModifier):
                self.send_message()
                return True  # Event handled
            
            # Shift+Enter -> insert new line (default behavior, just pass through)
            elif event.key() == Qt.Key_Return and (event.modifiers() & Qt.ShiftModifier):
                return False  # Let default handler insert newline
        
        return super().eventFilter(obj, event)
    
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
    
    def on_context_toggled(self, checked):
        """Handle context sharing toggle"""
        self.use_context = checked
        status = "enabled" if checked else "disabled"
        self.append_system(f"💬 Context sharing {status}")
    
    def on_model_changed(self, model_name):
        """Handle model selection change"""
        if not model_name or model_name == self.client.model_name:
            return
        
        old_model = self.client.model_name
        
        self.append_system(f"🔄 Switching: {old_model} → {model_name}")
        self.client.model_name = model_name
        self.client.ready = True
        self.setWindowTitle(f"Ollama Chat - {model_name}")
        self.append_system(f"✅ Now using: {model_name}")
        
        # Clear image attachment when switching models (text attachments stay)
        if self.attached_image_base64:
            self.clear_attachment()
            self.append_system("📎 Image attachment cleared (model switched)")
        
        if self.use_context and len(self.conversation_history) > 0:
            self.append_system(f"💬 Context shared: {len(self.conversation_history)//2} exchanges\n")
        else:
            self.append_system("💬 Fresh start (no context)\n")
    
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
            
            # Progress callback
            def on_progress(msg):
                self.signals.error_occurred.emit(f"PROGRESS:{msg}")
            
            # Completion callback
            def on_complete(success, error):
                # Remove progress line
                if hasattr(self, 'download_progress_pos'):
                    cursor = self.chat_display.textCursor()
                    cursor.setPosition(self.download_progress_pos)
                    cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
                    cursor.removeSelectedText()
                    delattr(self, 'download_progress_pos')
                
                if success:
                    self.append_system(f"✅ Successfully downloaded: {model_name}")
                    self.append_system("Refreshing model list...")
                    self.refresh_models()
                else:
                    self.append_error(f"Failed to download model:\n{error}")
                
                # Re-enable UI
                self.load_button.setEnabled(True)
                self.input_field.setEnabled(True)
                self.send_button.setEnabled(True)
            
            # Start async download
            self.client.pull_model_async(model_name, on_progress, on_complete)
    
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
        # Add initial thinking message and track its start position
        cursor = self.chat_display.textCursor()
        self.thinking_start_pos = cursor.position()
        self.chat_display.append(f"\n[ASSISTANT] Thinking")
        self.chat_display.moveCursor(QTextCursor.End)
        # Start timer (update every 500ms)
        self.thinking_timer.start(500)
    
    def update_thinking_animation(self):
        """Update the thinking animation dots"""
        self.thinking_dots = (self.thinking_dots + 1) % 4
        dots = "." * self.thinking_dots
        
        # Replace entire thinking line
        cursor = self.chat_display.textCursor()
        cursor.setPosition(self.thinking_start_pos)
        cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        cursor.removeSelectedText()
        
        # Re-add with animation
        self.chat_display.insertPlainText(f"\n[ASSISTANT] Thinking{dots}")
        self.chat_display.moveCursor(QTextCursor.End)
    
    def stop_thinking_animation(self):
        """Stop the thinking animation and remove it completely"""
        self.thinking_timer.stop()
        
        # Remove the entire thinking message
        if hasattr(self, 'thinking_start_pos'):
            cursor = self.chat_display.textCursor()
            cursor.setPosition(self.thinking_start_pos)
            cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
            cursor.removeSelectedText()
            delattr(self, 'thinking_start_pos')
    
    def attach_file(self):
        """Open file dialog to attach a file (image or text)"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select File",
            "",
            "All Files (*);;Images (*.png *.jpg *.jpeg *.gif *.bmp);;Text Files (*.txt *.md *.py *.json *.csv)"
        )
        
        if file_path:
            file_ext = Path(file_path).suffix.lower()
            
            # Check if it's an image
            if file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
                self._attach_image(file_path)
            # Check if it's a text file
            elif file_ext in ['.txt', '.md', '.py', '.js', '.json', '.csv', '.xml', '.html', '.css', '.yml', '.yaml']:
                self._attach_text(file_path)
            else:
                # Try to detect if it's text or binary
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        f.read(1024)  # Try to read as text
                    self._attach_text(file_path)
                except:
                    self.append_error(f"Unsupported file type: {file_ext}")
    
    def _attach_image(self, file_path):
        """Attach an image file"""
        try:
            # Read and encode image
            with open(file_path, 'rb') as f:
                image_data = f.read()
                self.attached_image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            self.attached_image_path = file_path
            self.attached_text_content = None
            self.attached_file_type = 'image'
            
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
                self.attachment_label.setScaledContents(False)
                self.attachment_label.show()
                self.clear_attach_button.show()
                self.append_system(f"📎 Attached image: {Path(file_path).name}")
            else:
                self.append_error(f"Could not load image: {file_path}")
                
        except Exception as e:
            self.append_error(f"Error attaching image: {e}")
    
    def _attach_text(self, file_path):
        """Attach a text file"""
        try:
            # Read text file
            with open(file_path, 'r', encoding='utf-8') as f:
                self.attached_text_content = f.read()
            
            self.attached_image_path = file_path
            self.attached_image_base64 = None
            self.attached_file_type = 'text'
            
            # Show text preview
            preview = self.attached_text_content[:200]
            if len(self.attached_text_content) > 200:
                preview += "..."
            
            self.attachment_label.setText(f"📄 {Path(file_path).name}\n{len(self.attached_text_content)} chars\n---\n{preview}")
            self.attachment_label.setWordWrap(True)
            self.attachment_label.show()
            self.clear_attach_button.show()
            self.append_system(f"📎 Attached text file: {Path(file_path).name} ({len(self.attached_text_content)} chars)")
            
        except Exception as e:
            self.append_error(f"Error attaching text file: {e}")
    
    def clear_attachment(self):
        """Clear attached file"""
        self.attached_image_path = None
        self.attached_image_base64 = None
        self.attached_text_content = None
        self.attached_file_type = None
        self.attachment_label.clear()
        self.attachment_label.hide()
        self.clear_attach_button.hide()
        self.append_system("📎 Attachment cleared")
    
    def send_message(self):
        """Send user message to Ollama"""
        if not self.client.ready:
            self.append_error("Ollama not ready")
            return
        
        message = self.input_field.toPlainText().strip()
        if not message and not self.attached_image_base64 and not self.attached_text_content:
            return
        
        # Build the full prompt
        full_prompt = message
        
        # Handle text attachments - include in prompt
        if self.attached_text_content:
            if not message:
                message = "Please analyze this file:"
            full_prompt = f"{message}\n\nFile content:\n```\n{self.attached_text_content}\n```"
        elif not message and self.attached_image_base64:
            # Default message if only image
            message = "What's in this image?"
            full_prompt = message
        
        # Store in conversation history
        self.conversation_history.append({
            'role': 'user',
            'content': message,
            'has_image': bool(self.attached_image_base64),
            'has_text': bool(self.attached_text_content)
        })
        
        # Clear input
        self.input_field.clear()
        
        # Display user message
        if self.attached_image_path and self.attached_file_type == 'image':
            self.append_user(f"{message} [📷 {Path(self.attached_image_path).name}]")
        elif self.attached_image_path and self.attached_file_type == 'text':
            self.append_user(f"{message} [📄 {Path(self.attached_image_path).name}]")
        else:
            self.append_user(message)
        
        # Start thinking animation
        self.start_thinking_animation()
        
        # Disable input while processing
        self.input_field.setEnabled(False)
        self.send_button.setEnabled(False)
        self.attach_button.setEnabled(False)
        self.model_combo.setEnabled(False)
        
        # Prepare images list (only for image attachments)
        images = [self.attached_image_base64] if self.attached_image_base64 else None
        
        # Vision models don't handle context well - NEVER send context with images
        if self.use_context and not self.attached_image_base64:
            prompt = self._build_context_prompt(full_prompt)
        else:
            prompt = full_prompt
        
        # Debug: log what we're sending
        print(f"[DEBUG] Sending to {self.client.model_name}:")
        print(f"[DEBUG] Prompt: {prompt[:100]}...")
        print(f"[DEBUG] Has image: {images is not None}")
        print(f"[DEBUG] Context used: {self.use_context and not self.attached_image_base64}")
        
        # Call async generate with callbacks
        self.client.generate_async(
            prompt=prompt,
            images=images,
            max_tokens=2000,  # Increased from 1000
            temperature=0.7,
            on_response=lambda resp: self.signals.response_received.emit(resp),
            on_error=lambda err: self.signals.error_occurred.emit(err)
        )
    
    def _build_context_prompt(self, current_message):
        """Build prompt with conversation context"""
        if len(self.conversation_history) <= 1:
            return current_message
        
        # Get last 10 exchanges (20 messages)
        recent = self.conversation_history[-20:]
        
        # Check if previous context had image descriptions
        has_image_context = any(msg.get('has_image') for msg in recent if msg['role'] == 'user')
        
        # Build a clear context
        if has_image_context:
            context = "You have access to the following image analysis from a previous conversation:\n\n"
        else:
            context = "Previous conversation:\n\n"
        
        for msg in recent[:-1]:  # Exclude current message
            if msg['role'] == 'user':
                if msg.get('has_image'):
                    context += f"Question about image: {msg['content']}\n"
                else:
                    context += f"User: {msg['content']}\n"
            else:
                context += f"Analysis: {msg['content']}\n\n"
        
        if has_image_context:
            context += f"\nBased on the image analysis above, {current_message}\n"
            context += "Answer directly using only the information provided above."
        else:
            context += f"\n{current_message}"
        
        return context
    
    def on_response(self, response):
        """Handle LLM response"""
        # Stop thinking animation
        self.stop_thinking_animation()
        
        # Debug: log response
        print(f"[DEBUG] Response received. Length: {len(response) if response else 0}")
        if response:
            print(f"[DEBUG] Response preview: {response[:100]}...")
        
        # Check if response is empty
        if not response or not response.strip():
            self.append_error("Model returned an empty response. Try rephrasing your question.")
            self.append_system("💡 Tip: Try more descriptive questions like 'Describe how many people are visible'")
            print(f"[DEBUG] Empty response received. Full response: '{response}'")
        else:
            # Store in conversation history
            self.conversation_history.append({
                'role': 'assistant',
                'content': response
            })
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

