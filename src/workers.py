"""
Background worker threads for ClaudeMood
"""
import subprocess
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


class ModelLoader(QThread):
    """Background worker for loading sentiment model"""
    finished = pyqtSignal(object)  # Emit loaded pipeline
    error = pyqtSignal(str)  # Emit error message

    def __init__(self, model_cache_dir):
        super().__init__()
        self.model_cache_dir = model_cache_dir

    def run(self):
        """Load the sentiment analysis model"""
        try:
            print("🤖 Loading RobBERT model in background...")

            # Set environment variables
            import os
            os.environ['TRANSFORMERS_CACHE'] = str(self.model_cache_dir)
            os.environ['HF_HOME'] = str(self.model_cache_dir)
            os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

            # Ensure cache directory exists
            self.model_cache_dir.mkdir(parents=True, exist_ok=True)

            # Load model and tokenizer
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    "DTAI-KULeuven/robbert-v2-dutch-sentiment",
                    cache_dir=str(self.model_cache_dir)
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    "DTAI-KULeuven/robbert-v2-dutch-sentiment",
                    cache_dir=str(self.model_cache_dir)
                )
                sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=model,
                    tokenizer=tokenizer
                )
                print("✅ Model loaded successfully!")
                self.finished.emit(sentiment_pipeline)
            except Exception as e:
                print(f"⚠️ Trying local cache: {e}")
                tokenizer = AutoTokenizer.from_pretrained(
                    "DTAI-KULeuven/robbert-v2-dutch-sentiment",
                    cache_dir=str(self.model_cache_dir),
                    local_files_only=True
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    "DTAI-KULeuven/robbert-v2-dutch-sentiment",
                    cache_dir=str(self.model_cache_dir),
                    local_files_only=True
                )
                sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=model,
                    tokenizer=tokenizer
                )
                print("✅ Model loaded from local cache!")
                self.finished.emit(sentiment_pipeline)

        except Exception as e:
            error_msg = f"Failed to load model: {e}"
            print(f"❌ {error_msg}")
            self.error.emit(error_msg)


class DataLoader(QThread):
    """Background worker for loading conversation data"""
    progress = pyqtSignal(str)  # Emit progress messages
    finished = pyqtSignal()  # Emit when done

    def __init__(self, app):
        super().__init__()
        self.app = app

    def run(self):
        """Load conversation data"""
        try:
            print("📊 Loading conversation data in background...")
            self.progress.emit("📊 Loading conversations...")
            self.app.load_recent_conversations()
            print("✅ Data loaded successfully!")
            self.finished.emit()
        except Exception as e:
            print(f"❌ Error loading data: {e}")


class AnalysisWorker(QThread):
    """Background worker for analyzing messages for a specific work day"""
    progress = pyqtSignal(int, int)  # Emit (current, total) progress
    finished = pyqtSignal(dict)  # Emit result data when done
    error = pyqtSignal(str)  # Emit error message

    def __init__(self, app, messages, work_day):
        super().__init__()
        self.app = app
        self.messages = messages
        self.work_day = work_day

    def run(self):
        """Analyze messages in background"""
        try:
            print(f"💪 Analyzing {len(self.messages)} messages in background...")

            sentiment_history = []
            breaks_today = []
            total = len(self.messages)

            for i, msg in enumerate(self.messages):
                # Emit progress every 10 messages
                if i % 10 == 0 or i == total - 1:
                    self.progress.emit(i + 1, total)

                # Analyze with sentiment analyzer
                prev_timestamp = self.messages[i-1]['timestamp'] if i > 0 else None

                # Call analyze_message_text with proper parameters
                result = self.app.analyze_single_message(
                    msg['text'],
                    msg['timestamp'],
                    conversation_file=msg.get('file', ''),
                    prev_global_timestamp=prev_timestamp
                )

                if result:
                    sentiment_history.append(result['message_data'])
                    if result.get('break_detected'):
                        breaks_today.append(result['break_data'])

            # Calculate stats
            cache_data = {
                'messages': sentiment_history,
                'breaks': breaks_today,
                'work_hours': self.calculate_work_hours(sentiment_history),
                'break_count': len(breaks_today),
                'message_count': len(sentiment_history),
                'avg_sentiment': sum(m['sentiment'] for m in sentiment_history) / len(sentiment_history) if sentiment_history else 0,
            }

            print(f"✅ Analysis complete! {len(sentiment_history)} messages analyzed")
            self.finished.emit(cache_data)

        except Exception as e:
            error_msg = f"Failed to analyze messages: {e}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            self.error.emit(error_msg)

    def calculate_work_hours(self, messages):
        """Calculate work hours from messages"""
        if not messages or len(messages) < 2:
            return 0

        first_time = messages[0]['timestamp']
        last_time = messages[-1]['timestamp']
        duration = (last_time - first_time).total_seconds() / 3600
        return max(0, duration)


class SearchWorker(QThread):
    """Background worker for searching conversation directories"""
    progress = pyqtSignal(str)  # Emit progress messages
    finished = pyqtSignal(list)  # Emit found directories

    def run(self):
        """Search for Claude conversations directories"""
        self.progress.emit("🔍 Searching for conversations directories...")

        found_dirs = []

        # Priority 1: Check for .claude/projects (Claude Code format)
        self.progress.emit("📂 Checking for Claude Code projects...")
        claude_projects = Path.home() / ".claude" / "projects"
        if claude_projects.exists() and claude_projects.is_dir():
            # Count JSONL files in subdirectories
            jsonl_count = sum(1 for _ in claude_projects.rglob("*.jsonl"))
            if jsonl_count > 0:
                found_dirs.append(claude_projects)
                self.progress.emit(f"✅ Found: .claude/projects ({jsonl_count} conversation files)")

        # Priority 2: Check standard Claude conversations locations
        self.progress.emit("📂 Checking standard locations...")
        search_paths = [
            Path.home() / "Library" / "Application Support" / "Claude",
            Path.home() / "Library" / "Application Support" / "Claude Code",
            Path.home() / ".config" / "Claude",
            Path.home() / ".config" / "Claude Code",
        ]

        for base_path in search_paths:
            if base_path.exists():
                conversations_dir = base_path / "conversations"
                if conversations_dir.exists() and conversations_dir.is_dir():
                    # Check if it has JSON files
                    json_files = list(conversations_dir.glob("*.json"))
                    if json_files:
                        found_dirs.append(conversations_dir)
                        self.progress.emit(f"✅ Found: {conversations_dir} ({len(json_files)} conversations)")

        # If still not found, do deeper search (slower)
        if not found_dirs:
            self.progress.emit("🔍 Doing deep search... (may take 10-20 seconds)")
            try:
                # Search for 'conversations' directories
                result = subprocess.run(
                    ['find', str(Path.home()), '-name', 'conversations', '-type', 'd', '-maxdepth', '5'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                for line in result.stdout.strip().split('\n'):
                    if line:
                        conv_dir = Path(line)
                        if conv_dir.exists():
                            json_files = list(conv_dir.glob("*.json"))
                            if json_files and len(json_files) > 0:
                                found_dirs.append(conv_dir)
                                self.progress.emit(f"✅ Found: {conv_dir} ({len(json_files)} conversations)")
            except subprocess.TimeoutExpired:
                self.progress.emit("⚠️  Search timeout after 30 seconds")
            except Exception as e:
                self.progress.emit(f"⚠️  Search error: {e}")

        if not found_dirs:
            self.progress.emit("❌ No conversations directories found")
        else:
            self.progress.emit(f"✅ Done! Found {len(found_dirs)} director{'y' if len(found_dirs) == 1 else 'ies'}")

        self.finished.emit(found_dirs)
