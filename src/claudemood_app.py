#!/usr/bin/env python3
"""
ClaudeMood - Real-time Developer Sentiment Tracker
Monitors Claude Code conversations and tracks your mood while coding
"""
import sys
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from daily_cache import DailyCache
from date_utils import DateUtils
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QProgressBar, QTextEdit, QTabWidget, QMessageBox,
    QDialog, QListWidget, QDialogButtonBox, QProgressDialog, QScrollArea
)
from PyQt6.QtCore import QTimer, Qt, QProcess, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from transformers import pipeline
import subprocess
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


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


def find_conversations_directories():
    """Search for Claude conversations directories (legacy sync version)"""
    print("🔍 Searching for conversations directories...")

    search_paths = [
        Path.home() / "Library" / "Application Support" / "Claude",
        Path.home() / "Library" / "Application Support" / "Claude Code",
        Path.home() / ".config" / "Claude",
        Path.home() / ".config" / "Claude Code",
    ]

    found_dirs = []

    # Quick search in common locations
    for base_path in search_paths:
        if base_path.exists():
            conversations_dir = base_path / "conversations"
            if conversations_dir.exists() and conversations_dir.is_dir():
                # Check if it has JSON files
                json_files = list(conversations_dir.glob("*.json"))
                if json_files:
                    found_dirs.append(conversations_dir)
                    print(f"✅ Found: {conversations_dir} ({len(json_files)} conversations)")

    # If not found, do deeper search (slower)
    if not found_dirs:
        print("🔍 Doing deep search (may take 10-20 seconds)...")
        try:
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
                            print(f"✅ Found: {conv_dir} ({len(json_files)} conversations)")
        except subprocess.TimeoutExpired:
            print("⚠️  Search timeout after 30 seconds")
        except Exception as e:
            print(f"⚠️  Search error: {e}")

    return found_dirs


def load_config():
    """Load configuration from config.json"""
    config_path = Path(__file__).parent.parent / "config.json"

    if not config_path.exists():
        print(f"⚠️  Config file not found: {config_path}")
        print("📋 Creating default config from template...")

        template_path = config_path.parent / "config.json.example"
        if template_path.exists():
            import shutil
            shutil.copy(template_path, config_path)
            print(f"✅ Config created: {config_path}")
        else:
            raise FileNotFoundError(f"Config template not found: {template_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Expand ~ in paths
    for key in ['conversations_dir', 'data_dir', 'export_dir', 'model_cache_dir']:
        if key in config:
            config[key] = Path(config[key]).expanduser()

    return config


def save_config(config):
    """Save configuration to config.json"""
    config_path = Path(__file__).parent.parent / "config.json"

    # Convert Path objects back to strings
    config_dict = config.copy()
    for key in ['conversations_dir', 'data_dir', 'export_dir', 'model_cache_dir']:
        if key in config_dict and isinstance(config_dict[key], Path):
            config_dict[key] = str(config_dict[key])

    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    print(f"💾 Config saved to: {config_path}")


class AppFileMonitor(FileSystemEventHandler):
    """Monitors ClaudeMood source files for changes (auto-restart)"""

    def __init__(self, callback):
        self.callback = callback
        self.last_modified = {}

    def on_modified(self, event):
        if event.is_directory:
            return

        # Only watch Python files and config
        if not (event.src_path.endswith('.py') or event.src_path.endswith('config.json')):
            return

        # Debounce: ignore rapid successive changes
        now = datetime.now()
        if event.src_path in self.last_modified:
            if (now - self.last_modified[event.src_path]).total_seconds() < 2:
                return

        self.last_modified[event.src_path] = now
        print(f"🔄 Source file changed: {event.src_path}")
        self.callback(event.src_path)

class DirectorySelectionDialog(QDialog):
    """Dialog for selecting conversations directory"""

    def __init__(self, directories, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Conversations Directory")
        self.setMinimumWidth(600)
        self.setMinimumHeight(300)

        self.selected_directory = None

        layout = QVBoxLayout()

        # Title
        title = QLabel("🔍 Multiple conversations directories found!\n\nSelect the correct directory:")
        title.setWordWrap(True)
        layout.addWidget(title)

        # Directory list
        self.list_widget = QListWidget()
        for directory in directories:
            dir_path = Path(directory)
            # Count conversation files (JSON or JSONL)
            if dir_path.name == "projects":
                file_count = sum(1 for _ in dir_path.rglob("*.jsonl"))
            else:
                file_count = len(list(dir_path.glob("*.json")))
            self.list_widget.addItem(f"{directory} ({file_count} conversations)")

        if directories:
            self.list_widget.setCurrentRow(0)

        layout.addWidget(self.list_widget)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)
        self.directories = directories

    def accept(self):
        """User clicked OK"""
        current_row = self.list_widget.currentRow()
        if current_row >= 0:
            self.selected_directory = self.directories[current_row]
        super().accept()


class SentimentChartWidget(QWidget):
    """Widget for displaying beautiful sentiment and work intensity charts"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Create matplotlib figure with 2 subplots
        self.figure = Figure(figsize=(10, 7), dpi=100, facecolor='#f8f9fa')
        self.canvas = FigureCanvas(self.figure)

        # Create layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Initialize empty charts
        self.ax1 = None  # Work intensity chart
        self.ax2 = None  # Sentiment chart
        self.update_chart([], [], [])

    def update_chart(self, timestamps, sentiments, breaks_today):
        """Update both sentiment and work intensity charts"""
        self.figure.clear()

        if not timestamps or len(timestamps) < 2:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No data yet...\nStart coding to see your activity!',
                    ha='center', va='center', fontsize=16, color='#6c757d',
                    weight='bold')
            ax.set_facecolor('#f8f9fa')
            ax.axis('off')
            self.canvas.draw()
            return

        # Create 2 subplots
        self.ax1 = self.figure.add_subplot(211)  # Work intensity (top)
        self.ax2 = self.figure.add_subplot(212)  # Sentiment (bottom)

        # === WORK INTENSITY CHART (TOP) ===
        self._plot_work_intensity(timestamps)

        # === SENTIMENT CHART (BOTTOM) ===
        self._plot_sentiment(timestamps, sentiments, breaks_today)

        # Tight layout
        self.figure.tight_layout(pad=2.0)
        self.canvas.draw()

    def _plot_work_intensity(self, timestamps):
        """Plot work intensity (messages per hour)"""
        # Calculate messages per 30-minute bucket
        from collections import defaultdict
        bucket_counts = defaultdict(int)

        for ts in timestamps:
            # Round to 30-minute bucket
            bucket = ts.replace(minute=(ts.minute // 30) * 30, second=0, microsecond=0)
            bucket_counts[bucket] += 1

        # Sort buckets
        buckets = sorted(bucket_counts.keys())
        counts = [bucket_counts[b] for b in buckets]

        if not buckets:
            return

        # Plot as bar chart with gradient
        bars = self.ax1.bar(buckets, counts, width=0.02, color='#667eea',
                            alpha=0.8, edgecolor='none')

        # Add gradient effect to bars
        for bar, count in zip(bars, counts):
            # Color intensity based on count
            intensity = min(count / max(counts), 1.0)
            bar.set_color(plt.cm.RdYlGn(0.3 + intensity * 0.5))

        # Styling
        self.ax1.set_facecolor('#ffffff')
        self.ax1.set_title('💪 Work Intensity (Messages per 30min)',
                          fontsize=14, fontweight='bold', color='#2c3e50', pad=15)
        self.ax1.set_ylabel('Messages', fontsize=11, color='#495057', fontweight='600')
        self.ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
        self.ax1.spines['top'].set_visible(False)
        self.ax1.spines['right'].set_visible(False)
        self.ax1.spines['left'].set_color('#dee2e6')
        self.ax1.spines['bottom'].set_color('#dee2e6')

        # Format x-axis
        self.ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        self.figure.autofmt_xdate(rotation=45)

        # Add max indicator
        max_count = max(counts)
        max_idx = counts.index(max_count)
        self.ax1.annotate(f'Peak: {max_count} msgs',
                         xy=(buckets[max_idx], max_count),
                         xytext=(10, 10), textcoords='offset points',
                         bbox=dict(boxstyle='round,pad=0.5', fc='#ffeaa7', alpha=0.8),
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='#fd79a8'),
                         fontsize=9, fontweight='bold')

    def _plot_sentiment(self, timestamps, sentiments, breaks_today):
        """Plot sentiment over time with beautiful gradients"""
        # Plot sentiment line with gradient fill
        x_numeric = mdates.date2num(timestamps)

        # Smooth the line with interpolation (if scipy available)
        try:
            if len(timestamps) > 10:
                from scipy.interpolate import make_interp_spline

                # Remove duplicates (scipy can't handle duplicate x values)
                unique_indices = []
                seen_times = set()
                for i, x in enumerate(x_numeric):
                    if x not in seen_times:
                        seen_times.add(x)
                        unique_indices.append(i)

                if len(unique_indices) > 3:  # Need at least 4 points for cubic spline
                    x_unique = [x_numeric[i] for i in unique_indices]
                    y_unique = [sentiments[i] for i in unique_indices]

                    x_smooth = np.linspace(x_unique[0], x_unique[-1], 300)
                    spl = make_interp_spline(x_unique, y_unique, k=3)
                    sentiments_smooth = spl(x_smooth)
                    timestamps_smooth = mdates.num2date(x_smooth)
                else:
                    # Not enough unique points for interpolation
                    timestamps_smooth = timestamps
                    sentiments_smooth = sentiments
            else:
                timestamps_smooth = timestamps
                sentiments_smooth = sentiments
        except (ImportError, ValueError) as e:
            # scipy not available or interpolation failed - use original data
            print(f"⚠️ Interpolation skipped: {e}")
            timestamps_smooth = timestamps
            sentiments_smooth = sentiments

        # Plot line with shadow effect
        self.ax2.plot(timestamps_smooth, sentiments_smooth, color='#5f27cd',
                     linewidth=3, label='Sentiment', zorder=5, alpha=0.9)

        # Fill area under curve with gradient
        positive_mask = np.array(sentiments_smooth) > 0
        negative_mask = np.array(sentiments_smooth) <= 0

        # Positive gradient (green)
        self.ax2.fill_between(timestamps_smooth, 0, sentiments_smooth,
                             where=positive_mask,
                             interpolate=True, alpha=0.3,
                             color='#00b894', label='Positive mood')

        # Negative gradient (red)
        self.ax2.fill_between(timestamps_smooth, 0, sentiments_smooth,
                             where=negative_mask,
                             interpolate=True, alpha=0.3,
                             color='#ff7675', label='Negative mood')

        # Visualize breaks as red zones
        if breaks_today:
            for brk in breaks_today:
                self.ax2.axvspan(brk['start'], brk['end'],
                                alpha=0.15, color='#d63031', zorder=1)
                # Add break label
                mid_time = brk['start'] + (brk['end'] - brk['start']) / 2
                self.ax2.text(mid_time, 0.9, f"☕ {brk['duration_minutes']:.0f}m",
                             ha='center', va='center', fontsize=8,
                             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8),
                             color='#d63031', fontweight='bold', zorder=10)

        # Add zero line
        self.ax2.axhline(y=0, color='#b2bec3', linestyle='--', linewidth=1.5, alpha=0.6, zorder=2)

        # Add trend line
        if len(x_numeric) >= 3:
            z = np.polyfit(x_numeric, sentiments, 1)
            p = np.poly1d(z)
            trend_line = p(x_numeric)
            trend_color = '#00b894' if z[0] > 0 else '#ff7675'
            self.ax2.plot(timestamps, trend_line, '--', color=trend_color,
                        linewidth=2, alpha=0.7, label=f'Trend {"↑" if z[0] > 0 else "↓"}', zorder=3)

        # Styling
        self.ax2.set_facecolor('#ffffff')
        self.ax2.set_title('😊 Sentiment Over Time',
                          fontsize=14, fontweight='bold', color='#2c3e50', pad=15)
        self.ax2.set_xlabel('Time', fontsize=11, color='#495057', fontweight='600')
        self.ax2.set_ylabel('Sentiment', fontsize=11, color='#495057', fontweight='600')
        self.ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.8, zorder=0)
        self.ax2.legend(loc='upper left', fontsize=9, framealpha=0.9, edgecolor='#dee2e6')
        self.ax2.set_ylim(-1.1, 1.1)

        # Remove top and right spines
        self.ax2.spines['top'].set_visible(False)
        self.ax2.spines['right'].set_visible(False)
        self.ax2.spines['left'].set_color('#dee2e6')
        self.ax2.spines['bottom'].set_color('#dee2e6')

        # Format x-axis
        self.ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        self.figure.autofmt_xdate(rotation=45)


class ConversationMonitor(FileSystemEventHandler):
    """Monitors Claude Code conversations for changes"""

    def __init__(self, callback, conversations_dir):
        self.callback = callback
        self.conversations_path = Path(conversations_dir)
        self.last_modified = {}
        self.pending_update = None

    def on_modified(self, event):
        if event.is_directory:
            return
        # Monitor both .json (old format) and .jsonl (new Claude Code format)
        if not (event.src_path.endswith('.json') or event.src_path.endswith('.jsonl')):
            return

        # Debounce: ignore rapid successive changes (1 second cooldown)
        now = datetime.now()
        if event.src_path in self.last_modified:
            if (now - self.last_modified[event.src_path]).total_seconds() < 1.0:
                return

        self.last_modified[event.src_path] = now
        self.callback(event.src_path)


class ClaudeMoodApp(QMainWindow):
    # Qt signal for thread-safe file change notifications
    conversation_changed_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        # Load configuration
        print("📋 Loading configuration...")
        self.config = load_config()
        print("✅ Config loaded!")

        # Set window properties from config
        ui_config = self.config.get('ui', {})
        width = ui_config.get('window_width', 900)
        height = ui_config.get('window_height', 700)

        self.setWindowTitle("ClaudeMood - Developer Sentiment Tracker 🎭")
        self.setGeometry(100, 100, width, height)

        # Data storage (initialize immediately)
        self.sentiment_history = []
        self.current_sentiment = 0.0
        self.daily_stats = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0})

        # Phase 1: Work health tracking
        self.breaks_today = []
        self.last_message_time = None
        self.last_reset_date = datetime.now().date()

        # Model loading state
        self.sentiment_analyzer = None
        self.model_loaded = False
        self.data_loaded = False

        # Daily cache system (prevents re-analyzing everything!)
        cache_dir = Path.home() / "ClaudeMood" / "cache"
        self.daily_cache = DailyCache(cache_dir)
        print(f"💾 Cache system initialized: {cache_dir}")

        # Date utilities with custom day boundaries
        day_start_hour = self.config.get('analysis', {}).get('day_start_hour', 4)
        self.date_utils = DateUtils(day_start_hour)
        print(f"📅 Day boundaries: {day_start_hour:02d}:00 - {day_start_hour:02d}:00")

        # Viewing state (which work day are we currently viewing)
        self.viewing_work_day = self.date_utils.get_current_work_day()
        self.all_messages_cache = []  # Cache of all messages for historical navigation

        # Connect signal to slot (thread-safe communication)
        self.conversation_changed_signal.connect(self.handle_conversation_changed)

        # Setup UI FIRST (so it appears immediately)
        self.init_ui()

        # Setup file monitoring
        self.setup_monitoring()

        # Setup auto-restart monitoring
        if self.config.get('auto_restart', {}).get('enabled', True):
            self.setup_auto_restart()

        # Update timer (refresh UI every N seconds from config)
        update_interval = ui_config.get('update_interval_seconds', 5) * 1000
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.refresh_ui)
        self.update_timer.start(update_interval)

        # Load model and data in background
        self.load_model_async()
        self.load_data_async()

    def load_model_async(self):
        """Load sentiment model in background thread"""
        model_cache_dir = self.config.get('model_cache_dir', Path.home() / "ClaudeMood" / "models" / "hf_cache")

        self.model_loader = ModelLoader(model_cache_dir)
        self.model_loader.finished.connect(self.on_model_loaded)
        self.model_loader.error.connect(self.on_model_error)
        self.model_loader.start()

        # Show loading state
        self.mood_label.setText("Loading model...")
        self.mood_emoji.setText("⏳")

    def on_model_loaded(self, sentiment_pipeline):
        """Called when model is loaded"""
        self.sentiment_analyzer = sentiment_pipeline
        self.model_loaded = True
        print("✅ Model ready for use!")

        # Update UI
        self.mood_label.setText("Model loaded - Waiting for data...")
        self.mood_emoji.setText("✅")

        # If data was already loaded, reanalyze all messages with the model
        # Note: sentiment_history may be empty because data was loaded without model
        if self.data_loaded:
            print("🔄 Reanalyzing messages with loaded model...")
            self.reanalyze_all_messages()

    def on_model_error(self, error_msg):
        """Called when model loading fails"""
        self.mood_label.setText(f"Error loading model: {error_msg}")
        self.mood_emoji.setText("❌")

    def load_data_async(self):
        """Load conversation data in background thread"""
        self.data_loader = DataLoader(self)
        self.data_loader.progress.connect(self.on_data_progress)
        self.data_loader.finished.connect(self.on_data_loaded)
        self.data_loader.start()

    def on_data_progress(self, message):
        """Called during data loading"""
        if hasattr(self, 'dashboard_summary'):
            self.dashboard_summary.setText(f"⏳ {message}")

    def on_data_loaded(self):
        """Called when data is loaded"""
        self.data_loaded = True
        print("✅ Data ready for display!")

        # Update all UI elements
        self.refresh_ui()

        # Update status
        if hasattr(self, 'mood_label'):
            self.update_mood_display()

    def init_ui(self):
        """Initialize the user interface"""
        # Apply modern stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f7fa;
            }
            QWidget {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            }
            QLabel {
                color: #2c3e50;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-size: 13px;
                font-weight: 600;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
            QTextEdit {
                background-color: white;
                border: 1px solid #dfe6e9;
                border-radius: 8px;
                padding: 12px;
                font-size: 12px;
                font-family: 'SF Mono', 'Monaco', 'Courier New', monospace;
                color: #2c3e50;
            }
            QTabWidget::pane {
                border: 1px solid #dfe6e9;
                border-radius: 8px;
                background-color: white;
                padding: 10px;
            }
            QTabBar::tab {
                background-color: #ecf0f1;
                color: #7f8c8d;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-weight: 600;
                font-size: 13px;
            }
            QTabBar::tab:selected {
                background-color: white;
                color: #3498db;
            }
            QTabBar::tab:hover {
                background-color: #e8ecef;
            }
            QProgressBar {
                border: 1px solid #dfe6e9;
                border-radius: 8px;
                text-align: center;
                height: 30px;
                background-color: #ecf0f1;
                font-weight: 600;
                font-size: 13px;
            }
            QProgressBar::chunk {
                border-radius: 7px;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        central_widget.setLayout(layout)

        # === HEADER ===
        header = QLabel("ClaudeMood")
        header.setFont(QFont("SF Pro Display", 26, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("""
            color: #2c3e50;
            padding: 15px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #667eea, stop:1 #764ba2);
            -webkit-background-clip: text;
            border-radius: 10px;
        """)
        layout.addWidget(header)

        # === DATE NAVIGATION BAR ===
        date_nav_widget = QWidget()
        date_nav_widget.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        date_nav_layout = QHBoxLayout()
        date_nav_layout.setContentsMargins(15, 10, 15, 10)
        date_nav_widget.setLayout(date_nav_layout)

        # Previous day button
        self.prev_day_btn = QPushButton("← Previous Day")
        self.prev_day_btn.setFont(QFont("SF Pro Text", 12))
        self.prev_day_btn.setStyleSheet("""
            QPushButton {
                background-color: #667eea;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 6px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #5568d3;
            }
            QPushButton:pressed {
                background-color: #4451b8;
            }
        """)
        self.prev_day_btn.clicked.connect(self.go_to_previous_day)
        date_nav_layout.addWidget(self.prev_day_btn)

        # Current date display
        self.viewing_date_label = QLabel("Today")
        self.viewing_date_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.viewing_date_label.setFont(QFont("SF Pro Text", 14, QFont.Weight.Bold))
        self.viewing_date_label.setStyleSheet("color: #2c3e50; padding: 5px;")
        date_nav_layout.addWidget(self.viewing_date_label, 1)  # stretch factor 1

        # Next day button
        self.next_day_btn = QPushButton("Next Day →")
        self.next_day_btn.setFont(QFont("SF Pro Text", 12))
        self.next_day_btn.setStyleSheet("""
            QPushButton {
                background-color: #667eea;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 6px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #5568d3;
            }
            QPushButton:pressed {
                background-color: #4451b8;
            }
            QPushButton:disabled {
                background-color: #cbd5e0;
                color: #a0aec0;
            }
        """)
        self.next_day_btn.clicked.connect(self.go_to_next_day)
        date_nav_layout.addWidget(self.next_day_btn)

        # "Today" quick button
        self.today_btn = QPushButton("Today")
        self.today_btn.setFont(QFont("SF Pro Text", 12))
        self.today_btn.setStyleSheet("""
            QPushButton {
                background-color: #48bb78;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 6px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #38a169;
            }
            QPushButton:pressed {
                background-color: #2f855a;
            }
        """)
        self.today_btn.clicked.connect(self.go_to_today)
        date_nav_layout.addWidget(self.today_btn)

        layout.addWidget(date_nav_widget)

        # === SUBTLE ALERT BANNER ===
        self.alert_banner = QLabel()
        self.alert_banner.setWordWrap(True)
        self.alert_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.alert_banner.setFont(QFont("SF Pro Text", 13, QFont.Weight.Bold))
        self.alert_banner.setStyleSheet("""
            background-color: #fff5f5;
            color: #e74c3c;
            padding: 10px;
            border-radius: 6px;
            border-left: 4px solid #e74c3c;
        """)
        self.alert_banner.hide()  # Hidden by default
        layout.addWidget(self.alert_banner)

        # === CURRENT MOOD DISPLAY ===
        mood_widget = QWidget()
        mood_widget.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 12px;
                border: 1px solid #e1e8ed;
            }
        """)
        mood_layout = QVBoxLayout()
        mood_layout.setSpacing(10)
        mood_layout.setContentsMargins(30, 20, 30, 20)
        mood_widget.setLayout(mood_layout)

        self.mood_emoji = QLabel("😐")
        self.mood_emoji.setFont(QFont("Apple Color Emoji", 64))
        self.mood_emoji.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mood_layout.addWidget(self.mood_emoji)

        self.mood_label = QLabel("Current Mood: Neutral (0.00)")
        self.mood_label.setFont(QFont("SF Pro Text", 18, QFont.Weight.DemiBold))
        self.mood_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mood_label.setStyleSheet("color: #34495e; padding: 5px;")
        mood_layout.addWidget(self.mood_label)

        self.mood_bar = QProgressBar()
        self.mood_bar.setMinimum(-100)
        self.mood_bar.setMaximum(100)
        self.mood_bar.setValue(0)
        self.mood_bar.setTextVisible(True)
        self.mood_bar.setFormat("%v% Mood Score")
        self.mood_bar.setFixedHeight(35)
        mood_layout.addWidget(self.mood_bar)

        layout.addWidget(mood_widget)

        # === TABS ===
        tabs = QTabWidget()

        # Tab 1: Dashboard (NEW!)
        dashboard_tab = QWidget()
        dashboard_layout = QVBoxLayout()
        dashboard_layout.setSpacing(15)
        dashboard_tab.setLayout(dashboard_layout)

        # Dashboard cards container
        cards_container = QWidget()
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(15)
        cards_container.setLayout(cards_layout)

        # Card 1: Work Hours
        work_card = QWidget()
        work_card.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 10px;
                border: 1px solid #e1e8ed;
                padding: 15px;
            }
        """)
        work_card_layout = QVBoxLayout()
        work_card.setLayout(work_card_layout)

        work_icon = QLabel("⏰")
        work_icon.setFont(QFont("Apple Color Emoji", 32))
        work_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        work_card_layout.addWidget(work_icon)

        self.dashboard_work_hours = QLabel("0.0h")
        self.dashboard_work_hours.setFont(QFont("SF Pro Display", 28, QFont.Weight.Bold))
        self.dashboard_work_hours.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dashboard_work_hours.setStyleSheet("color: #3498db;")
        work_card_layout.addWidget(self.dashboard_work_hours)

        work_label = QLabel("Work Hours")
        work_label.setFont(QFont("SF Pro Text", 12))
        work_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        work_label.setStyleSheet("color: #7f8c8d;")
        work_card_layout.addWidget(work_label)

        cards_layout.addWidget(work_card)

        # Card 2: Breaks
        break_card = QWidget()
        break_card.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 10px;
                border: 1px solid #e1e8ed;
                padding: 15px;
            }
        """)
        break_card_layout = QVBoxLayout()
        break_card.setLayout(break_card_layout)

        break_icon = QLabel("☕")
        break_icon.setFont(QFont("Apple Color Emoji", 32))
        break_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        break_card_layout.addWidget(break_icon)

        self.dashboard_breaks = QLabel("0")
        self.dashboard_breaks.setFont(QFont("SF Pro Display", 28, QFont.Weight.Bold))
        self.dashboard_breaks.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dashboard_breaks.setStyleSheet("color: #e67e22;")
        break_card_layout.addWidget(self.dashboard_breaks)

        break_label = QLabel("Breaks Taken")
        break_label.setFont(QFont("SF Pro Text", 12))
        break_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        break_label.setStyleSheet("color: #7f8c8d;")
        break_card_layout.addWidget(break_label)

        cards_layout.addWidget(break_card)

        # Card 3: Messages
        msg_card = QWidget()
        msg_card.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 10px;
                border: 1px solid #e1e8ed;
                padding: 15px;
            }
        """)
        msg_card_layout = QVBoxLayout()
        msg_card.setLayout(msg_card_layout)

        msg_icon = QLabel("💬")
        msg_icon.setFont(QFont("Apple Color Emoji", 32))
        msg_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        msg_card_layout.addWidget(msg_icon)

        self.dashboard_messages = QLabel("0")
        self.dashboard_messages.setFont(QFont("SF Pro Display", 28, QFont.Weight.Bold))
        self.dashboard_messages.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dashboard_messages.setStyleSheet("color: #9b59b6;")
        msg_card_layout.addWidget(self.dashboard_messages)

        msg_label = QLabel("Messages Today")
        msg_label.setFont(QFont("SF Pro Text", 12))
        msg_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        msg_label.setStyleSheet("color: #7f8c8d;")
        msg_card_layout.addWidget(msg_label)

        cards_layout.addWidget(msg_card)

        # Card 4: Health Status
        health_card = QWidget()
        health_card.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 10px;
                border: 1px solid #e1e8ed;
                padding: 15px;
            }
        """)
        health_card_layout = QVBoxLayout()
        health_card.setLayout(health_card_layout)

        self.dashboard_health_emoji = QLabel("✅")
        self.dashboard_health_emoji.setFont(QFont("Apple Color Emoji", 32))
        self.dashboard_health_emoji.setAlignment(Qt.AlignmentFlag.AlignCenter)
        health_card_layout.addWidget(self.dashboard_health_emoji)

        self.dashboard_health_status = QLabel("Healthy")
        self.dashboard_health_status.setFont(QFont("SF Pro Display", 20, QFont.Weight.Bold))
        self.dashboard_health_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dashboard_health_status.setStyleSheet("color: #27ae60;")
        health_card_layout.addWidget(self.dashboard_health_status)

        health_label = QLabel("Work Health")
        health_label.setFont(QFont("SF Pro Text", 12))
        health_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        health_label.setStyleSheet("color: #7f8c8d;")
        health_card_layout.addWidget(health_label)

        cards_layout.addWidget(health_card)

        dashboard_layout.addWidget(cards_container)

        # Dashboard summary section
        self.dashboard_summary = QLabel("Loading dashboard...")
        self.dashboard_summary.setWordWrap(True)
        self.dashboard_summary.setFont(QFont("SF Pro Text", 13))
        self.dashboard_summary.setStyleSheet("""
            QLabel {
                background-color: white;
                border-radius: 10px;
                border: 1px solid #e1e8ed;
                padding: 20px;
            }
        """)
        dashboard_layout.addWidget(self.dashboard_summary)

        tabs.addTab(dashboard_tab, "📊 Dashboard")

        # Tab 2: Live Feed
        live_tab = QWidget()
        live_layout = QVBoxLayout()
        live_tab.setLayout(live_layout)

        live_layout.addWidget(QLabel("Recent Messages:"))
        self.live_feed = QTextEdit()
        self.live_feed.setReadOnly(True)
        self.live_feed.setMaximumHeight(200)
        live_layout.addWidget(self.live_feed)

        tabs.addTab(live_tab, "💬 Live Feed")

        # Tab 3: Chart & Insights
        chart_tab = QWidget()
        chart_layout = QVBoxLayout()
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_tab.setLayout(chart_layout)

        # Time tracking info
        self.time_info_label = QLabel("Loading work session info...")
        self.time_info_label.setWordWrap(True)
        self.time_info_label.setFont(QFont("Arial", 11))
        self.time_info_label.setStyleSheet("padding: 10px; background-color: #f8f9fa; border-radius: 6px; margin: 5px;")
        chart_layout.addWidget(self.time_info_label)

        # Chart in scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #f8f9fa;
            }
        """)

        self.chart_widget = SentimentChartWidget()
        self.chart_widget.setMinimumHeight(800)  # Give charts room to breathe
        scroll_area.setWidget(self.chart_widget)
        chart_layout.addWidget(scroll_area)

        tabs.addTab(chart_tab, "📈 Chart & Insights")

        # Tab 4: Health (Phase 1)
        health_tab = QWidget()
        health_layout = QVBoxLayout()
        health_tab.setLayout(health_layout)

        self.health_label = QLabel("Loading health metrics...")
        self.health_label.setWordWrap(True)
        self.health_label.setFont(QFont("Arial", 11))
        health_layout.addWidget(self.health_label)

        tabs.addTab(health_tab, "🏥 Health")

        # Tab 5: Statistics
        stats_tab = QWidget()
        stats_layout = QVBoxLayout()
        stats_tab.setLayout(stats_layout)

        self.stats_label = QLabel("Loading statistics...")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)

        tabs.addTab(stats_tab, "📊 Statistics")

        # Tab 6: Alerts
        alerts_tab = QWidget()
        alerts_layout = QVBoxLayout()
        alerts_tab.setLayout(alerts_layout)

        self.alerts_label = QLabel("No alerts yet")
        self.alerts_label.setWordWrap(True)
        alerts_layout.addWidget(self.alerts_label)

        tabs.addTab(alerts_tab, "⚠️ Alerts")

        layout.addWidget(tabs)

        # === FOOTER BUTTONS ===
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        refresh_btn.clicked.connect(self.load_recent_conversations)
        button_layout.addWidget(refresh_btn)

        export_btn = QPushButton("💾 Export Data")
        export_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        export_btn.clicked.connect(self.export_data)
        button_layout.addWidget(export_btn)

        quit_btn = QPushButton("❌ Quit")
        quit_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        quit_btn.clicked.connect(self.close)
        button_layout.addWidget(quit_btn)

        layout.addLayout(button_layout)

    def setup_monitoring(self):
        """Setup file system monitoring for conversations"""
        conversations_dir = self.config.get('conversations_dir')

        # Check if directory exists
        if not conversations_dir.exists():
            print(f"⚠️  Conversations directory not found: {conversations_dir}")

            # Show search progress dialog
            progress = QProgressDialog("Searching for conversations directories...", "Cancel", 0, 0, self)
            progress.setWindowTitle("ClaudeMood - Searching...")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.setCancelButton(None)  # No cancel button during search
            progress.setAutoClose(False)
            progress.show()
            QApplication.processEvents()

            # Create and start search worker
            self.search_worker = SearchWorker()
            found_dirs = []

            def on_progress(message):
                progress.setLabelText(message)
                QApplication.processEvents()

            def on_finished(dirs):
                nonlocal found_dirs
                found_dirs = dirs
                progress.close()

            self.search_worker.progress.connect(on_progress)
            self.search_worker.finished.connect(on_finished)
            self.search_worker.start()

            # Wait for search to complete
            while self.search_worker.isRunning():
                QApplication.processEvents()

            if not found_dirs:
                QMessageBox.warning(
                    self,
                    "Directory Not Found",
                    f"Conversations directory not found:\n{conversations_dir}\n\n"
                    "Could not find any conversations directories automatically.\n\n"
                    "Please check your Claude Code installation."
                )
                return

            if len(found_dirs) == 1:
                # Auto-select if only one found
                conversations_dir = found_dirs[0]
                print(f"✅ Auto-selected: {conversations_dir}")

                # Update config
                self.config['conversations_dir'] = conversations_dir
                save_config(self.config)
                print("💾 Config updated with correct directory")

            else:
                # Multiple directories found - let user choose
                dialog = DirectorySelectionDialog(found_dirs, self)
                if dialog.exec() == QDialog.DialogCode.Accepted and dialog.selected_directory:
                    conversations_dir = Path(dialog.selected_directory)
                    print(f"✅ User selected: {conversations_dir}")

                    # Update config
                    self.config['conversations_dir'] = conversations_dir
                    save_config(self.config)
                    print("💾 Config updated with selected directory")
                else:
                    QMessageBox.warning(
                        self,
                        "No Directory Selected",
                        "No conversations directory selected. Monitoring disabled."
                    )
                    return

        # Start monitoring
        self.monitor = ConversationMonitor(self.on_conversation_changed, conversations_dir)
        self.observer = Observer()
        self.observer.schedule(self.monitor, str(conversations_dir), recursive=True)
        self.observer.start()
        print(f"✅ Monitoring conversations: {conversations_dir}")

    def setup_auto_restart(self):
        """Setup auto-restart when source files change"""
        project_root = Path(__file__).parent.parent
        watch_files = self.config.get('auto_restart', {}).get('watch_files', [])

        if not watch_files:
            print("⚠️  No files configured for auto-restart")
            return

        self.app_monitor = AppFileMonitor(self.on_source_file_changed)
        self.app_observer = Observer()

        # Watch src directory and config file
        self.app_observer.schedule(self.app_monitor, str(project_root / "src"), recursive=False)
        self.app_observer.schedule(self.app_monitor, str(project_root), recursive=False)
        self.app_observer.start()

        print(f"🔄 Auto-restart enabled - watching {len(watch_files)} files")

    def on_source_file_changed(self, file_path):
        """Called when source file changes - trigger restart"""
        reply = QMessageBox.question(
            self,
            "Restart Required",
            f"Source file changed:\n{Path(file_path).name}\n\nRestart ClaudeMood now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.restart_app()

    def on_conversation_changed(self, file_path):
        """Called when a conversation file changes (from file watcher thread)"""
        # Emit signal to schedule analysis on main thread (thread-safe)
        self.conversation_changed_signal.emit(file_path)

    def handle_conversation_changed(self, file_path):
        """Handle conversation changes on main thread (thread-safe)"""
        print(f"📝 Conversation changed: {file_path}")
        self.analyze_conversation(file_path)

    def reanalyze_all_messages(self):
        """Reanalyze all messages now that the model is loaded"""
        print("🔄 Model is now ready - reanalyzing all loaded messages...")
        # Simply reload all conversations with the model now available
        self.load_recent_conversations()
        # Force UI refresh
        self.refresh_ui()
        print("✅ Reanalysis complete!")

    def restart_app(self):
        """Restart the application"""
        print("🔄 Restarting ClaudeMood...")

        # Save current sentiment history before restart
        if self.sentiment_history:
            self.export_data(silent=True)

        # Close app and restart
        QProcess.startDetached(sys.executable, sys.argv)
        QApplication.quit()

    def load_recent_conversations(self):
        """Load and analyze recent conversations with caching (only analyze NEW messages!)"""
        conversations_dir = self.config.get('conversations_dir')

        if not conversations_dir.exists():
            self.live_feed.append("⚠️ Conversations directory not found")
            return

        today = datetime.now()
        today_date = today.date()

        # Try to load today's cache first
        print("💾 Checking cache for today...")
        cached_data = self.daily_cache.load_day_cache(today)

        if cached_data:
            print(f"✅ Loaded cache: {len(cached_data['messages'])} messages, {len(cached_data['breaks'])} breaks")
            # Load cached data
            self.sentiment_history = cached_data['messages']
            self.breaks_today = cached_data['breaks']
            if self.sentiment_history:
                self.current_sentiment = self.sentiment_history[-1]['sentiment']
        else:
            print("⚠️ No cache found - will analyze all messages")
            self.sentiment_history = []
            self.breaks_today = []

        # Get set of already-cached message IDs
        cached_message_ids = self.daily_cache.get_cached_message_ids(today)

        # Collect all user messages from recent conversations
        all_messages = []

        # Check if this is a projects directory (JSONL format) or conversations directory (JSON format)
        if conversations_dir.name == "projects":
            # Claude Code projects format: find all JSONL files in subdirectories
            jsonl_files = list(conversations_dir.rglob("*.jsonl"))
            if not jsonl_files:
                self.live_feed.append("⚠️ No conversation files found in projects")
                return

            # Sort by modification time, get most recent
            jsonl_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

            # Extract messages from ALL recent conversation files (not just 20 - we need all today's messages)
            for conv_file in jsonl_files[:100]:  # Check up to 100 files to get all today's messages
                try:
                    with open(conv_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                data = json.loads(line.strip())
                                message_obj = data.get('message', data)

                                if message_obj.get('role') == 'user':
                                    content = message_obj.get('content', '')
                                    timestamp_str = data.get('timestamp', '')

                                    # Handle content as list or string
                                    if isinstance(content, list):
                                        text_parts = []
                                        for part in content:
                                            if isinstance(part, dict) and part.get('type') == 'text':
                                                text_parts.append(part.get('text', ''))
                                        text = ' '.join(text_parts)
                                    else:
                                        text = content

                                    if text and len(text) >= 10:
                                        # Parse timestamp
                                        try:
                                            msg_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')).replace(tzinfo=None)
                                        except:
                                            msg_time = datetime.fromtimestamp(conv_file.stat().st_mtime)

                                        all_messages.append({
                                            'text': text,
                                            'timestamp': msg_time,
                                            'file': str(conv_file)
                                        })
                            except (json.JSONDecodeError, KeyError):
                                continue
                except Exception as e:
                    print(f"⚠️ Error reading {conv_file}: {e}")
                    continue

        else:
            # Old format: JSON files directly in conversations directory
            json_files = list(conversations_dir.glob("*.json"))
            if not json_files:
                self.live_feed.append("⚠️ No conversation files found")
                return

            # Sort by modification time, get most recent
            json_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

            # Extract messages from ALL recent conversations (to get all today's messages)
            for conv_file in json_files[:100]:  # Check up to 100 files
                try:
                    with open(conv_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        messages = data.get('chat', [])

                        for msg in messages:
                            if msg.get('sender') == 'human':
                                text = msg.get('text', '')
                                if text and len(text) >= 10:
                                    msg_time = datetime.fromtimestamp(conv_file.stat().st_mtime)
                                    all_messages.append({
                                        'text': text,
                                        'timestamp': msg_time,
                                        'file': str(conv_file)
                                    })
                except Exception as e:
                    print(f"⚠️ Error reading {conv_file}: {e}")
                    continue

        # Sort all messages by timestamp (CRITICAL for correct break detection!)
        all_messages.sort(key=lambda m: m['timestamp'])

        # Store ALL messages for historical navigation
        self.all_messages_cache = all_messages

        # Use DateUtils to filter messages for current viewing work day
        current_work_day = self.date_utils.get_work_day_for_timestamp(today)
        today_messages = self.date_utils.get_all_messages_for_work_day(all_messages, current_work_day)

        # Get recent 25 for display
        recent_messages = all_messages[-25:]

        print(f"📊 Loaded {len(all_messages)} total messages, {len(today_messages)} from current work day")

        # If model not loaded yet, just count the messages and stop
        # The reanalyze will happen when model is ready
        if not self.sentiment_analyzer:
            print("⏳ Model not ready yet - will analyze messages after model loads")
            return

        # Filter NEW messages (not in cache)
        new_messages = []
        for msg in today_messages:
            msg_id = f"{msg['timestamp']}_{msg['text'][:50]}"
            if msg_id not in cached_message_ids:
                new_messages.append(msg)

        if new_messages:
            print(f"🆕 Found {len(new_messages)} new messages to analyze (skipping {len(today_messages) - len(new_messages)} cached)")

            # Analyze only NEW messages (incrementally!)
            for msg in new_messages:
                # Use the last message time from history (or cache) for break detection
                prev_timestamp = None
                if self.sentiment_history:
                    prev_timestamp = self.sentiment_history[-1]['timestamp']

                self.analyze_message_text(
                    msg['text'],
                    msg['timestamp'],
                    conversation_file=msg.get('file', ''),
                    prev_global_timestamp=prev_timestamp
                )

            print(f"💪 Analyzed {len(new_messages)} new messages!")
        else:
            print(f"✅ All messages already in cache! (Total: {len(today_messages)})")

        # Save updated cache
        cache_data = {
            'messages': self.sentiment_history,
            'breaks': self.breaks_today,
            'work_hours': self.calculate_todays_work_hours() if self.sentiment_history else 0,
            'break_count': len(self.breaks_today),
            'message_count': len(self.sentiment_history),
            'avg_sentiment': sum(m['sentiment'] for m in self.sentiment_history) / len(self.sentiment_history) if self.sentiment_history else 0,
        }
        self.daily_cache.save_day_cache(today, cache_data)

        # Update UI to show current viewing date
        self.update_viewing_date_label()

    def check_daily_reset(self, current_timestamp):
        """Check if we need to reset daily tracking (new day)"""
        current_date = current_timestamp.date()

        if current_date > self.last_reset_date:
            # Save yesterday's snapshot before resetting
            if self.sentiment_history:
                self.save_daily_snapshot()

            # Reset daily tracking
            print(f"📅 New day detected - resetting daily tracking")
            self.breaks_today = []
            self.last_message_time = None
            self.last_reset_date = current_date

    def analyze_message_text(self, text, timestamp, conversation_file='', prev_global_timestamp=None):
        """Analyze sentiment of a single message text

        Args:
            text: Message text to analyze
            timestamp: When the message was sent
            conversation_file: Which conversation file this came from (for parallel session tracking)
            prev_global_timestamp: Timestamp of previous message in GLOBAL timeline (not per-conversation)
        """
        try:
            if not text or len(text) < 10:
                return

            # Check if model is loaded
            if not self.sentiment_analyzer:
                print("⚠️ Model not loaded yet, skipping sentiment analysis")
                return

            # Phase 1: Check for daily reset
            self.check_daily_reset(timestamp)

            # Phase 1: Detect break before this message (across ALL conversations!)
            # Use prev_global_timestamp if provided (batch analysis), otherwise use self.last_message_time (real-time)
            break_info = self.detect_break(timestamp, prev_global_timestamp=prev_global_timestamp)

            # Analyze sentiment (use max_text_length from config)
            max_length = self.config.get('analysis', {}).get('max_text_length', 512)
            result = self.sentiment_analyzer(text[:max_length])[0]
            label = result['label'].lower()
            score = result['score']

            # Convert to -1 to 1 scale
            if label == 'positive':
                sentiment = score
            elif label == 'negative':
                sentiment = -score
            else:
                sentiment = 0.0

            # Update stats
            self.sentiment_history.append({
                'timestamp': timestamp,
                'sentiment': sentiment,
                'text': text[:100],
                'break_before': break_info is not None,  # Phase 1: Track if there was a break before this message
                'conversation_file': conversation_file  # Track which conversation this was in
            })

            # Update current sentiment (rolling average from config)
            window_size = self.config.get('analysis', {}).get('rolling_average_window', 10)
            recent = [h['sentiment'] for h in self.sentiment_history[-window_size:]]
            self.current_sentiment = sum(recent) / len(recent)

            # Update UI - limit to last 25 messages
            self.update_live_feed()
            self.update_mood_display()

        except Exception as e:
            print(f"❌ Error analyzing message: {e}")

    def get_mood_emoji(self, sentiment):
        """Get mood emoji for a sentiment value"""
        thresholds = self.config.get('mood_thresholds', {})

        very_positive = thresholds.get('very_positive', 0.3)
        positive = thresholds.get('positive', 0.1)
        neutral_low = thresholds.get('neutral_low', -0.1)
        negative = thresholds.get('negative', -0.3)

        if sentiment > very_positive:
            return "😄"
        elif sentiment > positive:
            return "🙂"
        elif sentiment > neutral_low:
            return "😐"
        elif sentiment > negative:
            return "😕"
        else:
            return "😞"

    def update_live_feed(self):
        """Update Live Feed with last 25 messages"""
        # Get last 25 messages from history
        recent = self.sentiment_history[-25:]

        # Clear and rebuild feed
        self.live_feed.clear()
        for entry in recent:
            timestamp = entry['timestamp']
            sentiment = entry['sentiment']
            text = entry['text']
            mood_emoji = self.get_mood_emoji(sentiment)
            break_indicator = " ☕" if entry.get('break_before', False) else ""

            self.live_feed.append(f"[{timestamp.strftime('%H:%M')}] {mood_emoji} {sentiment:+.2f}{break_indicator}: {text[:80]}")

    def analyze_conversation(self, file_path):
        """Analyze sentiment of a conversation (supports both JSON and JSONL formats)"""
        try:
            # Check if model is loaded
            if not self.sentiment_analyzer:
                return

            file_path_obj = Path(file_path)
            text = None

            # Handle JSONL format (Claude Code projects)
            if file_path_obj.suffix == '.jsonl':
                with open(file_path, 'r', encoding='utf-8') as f:
                    last_user_message = None
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            message_obj = data.get('message', data)

                            if message_obj.get('role') == 'user':
                                content = message_obj.get('content', '')

                                # Handle content as list or string
                                if isinstance(content, list):
                                    text_parts = []
                                    for part in content:
                                        if isinstance(part, dict) and part.get('type') == 'text':
                                            text_parts.append(part.get('text', ''))
                                    last_user_message = ' '.join(text_parts)
                                else:
                                    last_user_message = content
                        except (json.JSONDecodeError, KeyError):
                            continue

                    text = last_user_message

            # Handle JSON format (old conversations directory)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                messages = data.get('chat', [])
                if not messages:
                    return

                # Get last message
                last_msg = messages[-1]
                text = last_msg.get('text', '')
                sender = last_msg.get('sender', 'unknown')

                if sender != 'human':
                    return

            # Skip if no text found or text too short
            if not text or len(text) < 10:
                return

            # Use current time as timestamp for real-time messages
            timestamp = datetime.now()

            # Note: Break detection is skipped here to avoid incorrect breaks
            # Breaks are calculated properly in load_recent_conversations()
            # where ALL messages are sorted chronologically across conversations

            # Analyze sentiment (use max_text_length from config)
            max_length = self.config.get('analysis', {}).get('max_text_length', 512)
            result = self.sentiment_analyzer(text[:max_length])[0]
            label = result['label'].lower()
            score = result['score']

            # Convert to -1 to 1 scale
            if label == 'positive':
                sentiment = score
            elif label == 'negative':
                sentiment = -score
            else:
                sentiment = 0.0

            # Update stats (real-time monitoring - breaks will be recalculated on next load)
            self.sentiment_history.append({
                'timestamp': timestamp,
                'sentiment': sentiment,
                'text': text[:100],
                'break_before': False,  # Will be set correctly by load_recent_conversations()
                'conversation_file': str(file_path)  # Track which conversation this was in
            })

            # Update current sentiment (rolling average from config)
            window_size = self.config.get('analysis', {}).get('rolling_average_window', 10)
            recent = [h['sentiment'] for h in self.sentiment_history[-window_size:]]
            self.current_sentiment = sum(recent) / len(recent)

            # Update UI - limit to last 25 messages
            self.update_live_feed()
            self.update_mood_display()

        except Exception as e:
            print(f"❌ Error analyzing {file_path}: {e}")

    def update_mood_display(self):
        """Update the mood display based on current sentiment"""
        sentiment = self.current_sentiment
        thresholds = self.config.get('mood_thresholds', {})

        # Get thresholds from config
        very_positive = thresholds.get('very_positive', 0.3)
        positive = thresholds.get('positive', 0.1)
        neutral_low = thresholds.get('neutral_low', -0.1)
        negative = thresholds.get('negative', -0.3)

        # Update emoji based on thresholds
        if sentiment > very_positive:
            emoji = "😄"
            mood = "Happy"
            color = "#4CAF50"
        elif sentiment > positive:
            emoji = "🙂"
            mood = "Good"
            color = "#8BC34A"
        elif sentiment > neutral_low:
            emoji = "😐"
            mood = "Neutral"
            color = "#FFC107"
        elif sentiment > negative:
            emoji = "😕"
            mood = "Stressed"
            color = "#FF9800"
        else:
            emoji = "😞"
            mood = "Frustrated"
            color = "#F44336"

        self.mood_emoji.setText(emoji)
        self.mood_label.setText(f"Current Mood: {mood} ({sentiment:+.2f})")
        self.mood_label.setStyleSheet(f"color: {color};")

        # Update progress bar
        bar_value = int(sentiment * 100)
        self.mood_bar.setValue(bar_value)

        # Color the progress bar with gradient
        if sentiment > 0.1:
            # Positive gradient
            gradient_color = f"""
                QProgressBar::chunk {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 {color}, stop:1 #a8e6cf);
                    border-radius: 7px;
                }}
            """
        elif sentiment < -0.1:
            # Negative gradient
            gradient_color = f"""
                QProgressBar::chunk {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #ff6b6b, stop:1 {color});
                    border-radius: 7px;
                }}
            """
        else:
            # Neutral
            gradient_color = f"""
                QProgressBar::chunk {{
                    background-color: {color};
                    border-radius: 7px;
                }}
            """

        self.mood_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 2px solid #dfe6e9;
                border-radius: 10px;
                text-align: center;
                height: 35px;
                background-color: #ecf0f1;
                font-weight: bold;
                font-size: 14px;
                color: #2c3e50;
            }}
            {gradient_color}
        """)

    def get_todays_messages(self):
        """Get all messages from today"""
        today = datetime.now().date()
        return [h for h in self.sentiment_history if h['timestamp'].date() == today]

    def calculate_work_session_info(self):
        """Calculate work session start time, duration, and trend"""
        todays_messages = self.get_todays_messages()

        if not todays_messages:
            return None, None, None

        # Start time = first message today
        start_time = todays_messages[0]['timestamp']

        # Duration = now - start time
        duration = datetime.now() - start_time
        hours = int(duration.total_seconds() // 3600)
        minutes = int((duration.total_seconds() % 3600) // 60)

        # Trend calculation (slope of sentiment over time)
        if len(todays_messages) >= 3:
            timestamps_numeric = mdates.date2num([m['timestamp'] for m in todays_messages])
            sentiments = [m['sentiment'] for m in todays_messages]
            z = np.polyfit(timestamps_numeric, sentiments, 1)
            trend_slope = z[0]

            # Interpret trend
            if trend_slope > 0.01:
                trend = "Improving ↗"
                trend_color = "green"
            elif trend_slope < -0.01:
                trend = "Declining ↘"
                trend_color = "red"
            else:
                trend = "Stable →"
                trend_color = "orange"
        else:
            trend = "Not enough data"
            trend_color = "gray"

        return (start_time, f"{hours}h {minutes}m", trend, trend_color)

    def update_chart_and_insights(self):
        """Update the chart and time tracking info"""
        todays_messages = self.get_todays_messages()

        if todays_messages:
            timestamps = [m['timestamp'] for m in todays_messages]
            sentiments = [m['sentiment'] for m in todays_messages]
            # Pass breaks_today to the chart for visualization
            self.chart_widget.update_chart(timestamps, sentiments, self.breaks_today)

            # Update time tracking
            start_time, duration, trend, trend_color = self.calculate_work_session_info()

            info_text = f"""
📅 Today's Session:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🕐 Started: {start_time.strftime('%H:%M')}
⏱️  Duration: {duration}
📊 Messages: {len(todays_messages)}
📈 Trend: <span style="color: {trend_color}; font-weight: bold;">{trend}</span>
"""
            self.time_info_label.setText(info_text)
        else:
            self.time_info_label.setText("⏳ No messages yet today. Start coding to see your session!")

    def update_dashboard(self):
        """Update dashboard metrics"""
        todays_messages = self.get_todays_messages()
        health_status = self.calculate_work_health_status()

        # Update metric cards
        work_hours = health_status.get('work_hours', 0)
        break_count = health_status.get('break_count', 0)

        self.dashboard_work_hours.setText(f"{work_hours:.1f}h")
        self.dashboard_breaks.setText(str(break_count))
        self.dashboard_messages.setText(str(len(todays_messages)))

        # Update health status card
        status_label = health_status.get('label', '⏳ No Data')
        status_color = health_status.get('color', '#999999')

        # Map status to emoji
        status_emoji_map = {
            'no_data': '⏳',
            'healthy': '✅',
            'overwork': '⚠️',
            'extreme': '🔴',
            'late_night': '🌙',
            'no_breaks': '⛔'
        }
        status_emoji = status_emoji_map.get(health_status.get('status', 'no_data'), '✅')

        self.dashboard_health_emoji.setText(status_emoji)
        self.dashboard_health_status.setText(status_label.split()[1] if ' ' in status_label else status_label)
        self.dashboard_health_status.setStyleSheet(f"color: {status_color};")

        # Update dashboard summary
        if not todays_messages:
            summary_text = "⏳ No work session data yet today. Start coding to see your metrics!"
        else:
            start_time = todays_messages[0]['timestamp']
            avg_sentiment = sum(m['sentiment'] for m in todays_messages) / len(todays_messages)

            # Calculate total break time
            total_break_minutes = sum(b['duration_minutes'] for b in self.breaks_today) if self.breaks_today else 0

            # Calculate parallel sessions
            current_parallel, peak_parallel = self.calculate_parallel_sessions()

            warnings = health_status.get('warnings', [])

            summary_text = f"""
<b>📅 Today's Summary</b><br><br>

<b>Session Started:</b> {start_time.strftime('%H:%M')}<br>
<b>Net Work Time:</b> {work_hours:.1f} hours<br>
<b>Total Break Time:</b> {total_break_minutes:.0f} minutes ({len(self.breaks_today)} breaks)<br>
<b>Average Sentiment:</b> {avg_sentiment:+.2f} {'😊' if avg_sentiment > 0.1 else '😐' if avg_sentiment > -0.1 else '😞'}<br>
<b>Active Sessions:</b> {current_parallel} currently | Peak: {peak_parallel} parallel<br><br>
"""

            if warnings:
                summary_text += "<b>⚠️ Health Alerts:</b><br>"
                for warning in warnings[:3]:  # Show max 3 warnings
                    summary_text += f"• {warning}<br>"
            else:
                summary_text += "<b>✅ All health metrics are good!</b>"

        self.dashboard_summary.setText(summary_text)

    def refresh_ui(self):
        """Refresh UI elements"""
        # Only update if data is loaded
        if not self.data_loaded or not self.sentiment_history:
            return

        self.update_dashboard()  # NEW!
        self.update_statistics()
        self.update_health_metrics()  # Phase 1
        self.check_alerts()
        self.update_chart_and_insights()

    def update_statistics(self):
        """Update statistics display"""
        if not self.sentiment_history:
            return

        total = len(self.sentiment_history)
        positive = len([h for h in self.sentiment_history if h['sentiment'] > 0.1])
        negative = len([h for h in self.sentiment_history if h['sentiment'] < -0.1])
        neutral = total - positive - negative

        avg_sentiment = sum(h['sentiment'] for h in self.sentiment_history) / total

        # Phase 1: Add break and work hour statistics
        todays_messages = self.get_todays_messages()
        work_hours = self.calculate_todays_work_hours() if todays_messages else 0
        break_count = len(self.breaks_today)

        stats_text = f"""
📊 Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📨 Message Statistics:
Total Messages: {total}
Today's Messages: {len(todays_messages)}
Average Sentiment: {avg_sentiment:+.3f}

😄 Positive: {positive} ({positive/total*100:.1f}%)
😐 Neutral: {neutral} ({neutral/total*100:.1f}%)
😞 Negative: {negative} ({negative/total*100:.1f}%)

Current Mood Score: {self.current_sentiment:+.3f}

⏰ Work & Break Statistics (Today):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Work Hours: {work_hours:.1f}h (net)
Breaks Taken: {break_count}
"""

        if self.breaks_today:
            total_break_time = sum(b['duration_minutes'] for b in self.breaks_today)
            avg_break = total_break_time / len(self.breaks_today)
            stats_text += f"Total Break Time: {total_break_time:.0f} min ({total_break_time/60:.1f}h)\n"
            stats_text += f"Average Break: {avg_break:.0f} min\n"

        self.stats_label.setText(stats_text)

    def update_health_metrics(self):
        """Update health metrics display (Phase 1)"""
        health_status = self.calculate_work_health_status()
        todays_messages = self.get_todays_messages()

        if not todays_messages:
            self.health_label.setText("⏳ No work session data yet today.")
            return

        work_hours = health_status.get('work_hours', 0)
        break_count = health_status.get('break_count', 0)
        status_label = health_status.get('label', 'No Data')
        status_color = health_status.get('color', '#999999')
        warnings = health_status.get('warnings', [])

        # Calculate break statistics
        if self.breaks_today:
            avg_break_duration = sum(b['duration_minutes'] for b in self.breaks_today) / len(self.breaks_today)
            total_break_time = sum(b['duration_minutes'] for b in self.breaks_today)
        else:
            avg_break_duration = 0
            total_break_time = 0

        # Get start and end times
        start_time = todays_messages[0]['timestamp']
        last_message_time = todays_messages[-1]['timestamp']

        # Check if currently working (last message within 30 min)
        is_currently_working = (datetime.now() - last_message_time).total_seconds() < 1800

        health_text = f"""
🏥 WORK HEALTH STATUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

<span style="color: {status_color}; font-size: 14pt; font-weight: bold;">{status_label}</span>

📊 Today's Metrics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⏰ Work Hours: {work_hours:.1f}h (net, excluding breaks)
🕐 Started: {start_time.strftime('%H:%M')}
🕒 Last Active: {last_message_time.strftime('%H:%M')}
{'🟢 Currently Working' if is_currently_working else '⚪ Session Paused'}

☕ Break Analysis:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Break Count: {break_count}
Total Break Time: {total_break_time:.0f} minutes ({total_break_time/60:.1f}h)
Average Break Duration: {avg_break_duration:.0f} minutes
"""

        # Add break details if any
        if self.breaks_today:
            health_text += "\nBreak Schedule:\n"
            for i, brk in enumerate(self.breaks_today, 1):
                start = brk['start'].strftime('%H:%M')
                end = brk['end'].strftime('%H:%M')
                duration = brk['duration_minutes']
                health_text += f"  {i}. {start} - {end} ({duration:.0f} min)\n"

        # Add warnings
        if warnings:
            health_text += "\n⚠️  Health Warnings:\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            for warning in warnings:
                health_text += f"{warning}\n"
        else:
            health_text += "\n✅ No health warnings - Keep up the good work!\n"

        # Add recommendations
        health_config = self.config.get('health', {})
        healthy_max = health_config.get('healthy_max_hours', 8)
        min_breaks = health_config.get('min_breaks_per_day', 2)

        health_text += "\n💡 Recommendations:\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        if work_hours < healthy_max and break_count >= min_breaks:
            health_text += "✅ You're maintaining a healthy work pattern!\n"
        else:
            if work_hours >= healthy_max:
                health_text += f"⚠️  Consider wrapping up - you've worked {work_hours:.1f}h today\n"
            if break_count < min_breaks:
                health_text += f"☕ Take more breaks - aim for at least {min_breaks} per day\n"

        self.health_label.setText(health_text)

    def check_alerts(self):
        """Check for mood and health alerts (Phase 1)"""
        if len(self.sentiment_history) < 5:
            return

        alert_config = self.config.get('alerts', {})
        high_stress = alert_config.get('high_stress_threshold', -0.3)
        very_negative = alert_config.get('very_negative_threshold', -0.5)
        consecutive_count = alert_config.get('consecutive_negative_count', 4)

        recent = self.sentiment_history[-5:]
        recent_sentiments = [h['sentiment'] for h in recent]
        avg_recent = sum(recent_sentiments) / len(recent_sentiments)

        alerts = []
        mood_alerts = []
        health_alerts = []

        # Mood alerts (SHORT!)
        if avg_recent < high_stress:
            mood_alerts.append("⚠️ High stress")

        if len([s for s in recent_sentiments if s < -0.2]) >= consecutive_count:
            mood_alerts.append("⚠️ Multiple negative messages")

        if self.current_sentiment < very_negative:
            mood_alerts.append("🚨 Very negative mood")

        # Phase 1: Health-based alerts
        notifications_config = self.config.get('notifications', {})
        if notifications_config.get('enabled', True):
            health_status = self.calculate_work_health_status()
            todays_messages = self.get_todays_messages()

            if todays_messages:
                work_hours = health_status.get('work_hours', 0)
                break_count = health_status.get('break_count', 0)

                # Late night warning
                if notifications_config.get('late_night_warnings', True):
                    current_hour = datetime.now().hour
                    late_night_hour = self.config.get('health', {}).get('late_night_hour', 22)
                    if current_hour >= late_night_hour:
                        last_message_time = todays_messages[-1]['timestamp']
                        if (datetime.now() - last_message_time).total_seconds() < 1800:
                            health_alerts.append(f"🌙 Late night ({late_night_hour}:00+)")

                # Daily limit alert
                if notifications_config.get('daily_limit_alerts', True):
                    overwork_threshold = self.config.get('health', {}).get('overwork_threshold', 8)
                    extreme_threshold = self.config.get('health', {}).get('extreme_threshold', 10)

                    if work_hours > extreme_threshold:
                        health_alerts.append(f"🔴 {work_hours:.1f}h worked (extreme!)")
                    elif work_hours > overwork_threshold:
                        health_alerts.append(f"⚠️ {work_hours:.1f}h worked")

                # Break reminder
                if notifications_config.get('break_reminders', True):
                    min_breaks = self.config.get('health', {}).get('min_breaks_per_day', 2)
                    if work_hours > 4 and break_count < min_breaks:
                        health_alerts.append(f"☕ {break_count} breaks only")

                    # Check time since last break
                    if self.last_message_time:
                        time_since_last = (datetime.now() - self.last_message_time).total_seconds() / 60
                        max_gap_minutes = self.config.get('health', {}).get('max_gap_minutes', 240)
                        if time_since_last > max_gap_minutes and time_since_last < max_gap_minutes + 30:
                            health_alerts.append(f"⏰ {time_since_last/60:.1f}h straight")

        # Combine alerts
        if mood_alerts:
            alerts.append("😞 MOOD ALERTS:\n━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" + "\n".join(mood_alerts))

        if health_alerts:
            alerts.append("\n🏥 HEALTH ALERTS:\n━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" + "\n".join(health_alerts))

        if alerts:
            # Update alerts tab
            self.alerts_label.setText("\n".join(alerts))
            self.alerts_label.setStyleSheet("color: #F44336; font-weight: bold;")

            # Show subtle banner (NO BLINKING!)
            all_alerts = mood_alerts + health_alerts
            banner_text = " • ".join(all_alerts)  # Simple separator
            self.alert_banner.setText(banner_text)
            self.alert_banner.setStyleSheet("""
                background-color: #fff5f5;
                color: #e74c3c;
                padding: 10px;
                border-radius: 6px;
                border-left: 4px solid #e74c3c;
            """)
            self.alert_banner.show()
        else:
            # No alerts - hide banner
            self.alert_banner.hide()

            self.alerts_label.setText("✅ All good! No alerts.\n\n💪 Keep coding with healthy habits!")
            self.alerts_label.setStyleSheet("color: #4CAF50;")

    # === DATE NAVIGATION FUNCTIONS ===

    def go_to_previous_day(self):
        """Navigate to the previous work day"""
        self.viewing_work_day = self.date_utils.get_previous_work_day(self.viewing_work_day)
        self.load_work_day_data(self.viewing_work_day)

    def go_to_next_day(self):
        """Navigate to the next work day"""
        new_day = self.date_utils.get_next_work_day(self.viewing_work_day)
        current_day = self.date_utils.get_current_work_day()

        # Don't go beyond today
        if new_day <= current_day:
            self.viewing_work_day = new_day
            self.load_work_day_data(self.viewing_work_day)

    def go_to_today(self):
        """Jump back to today"""
        self.viewing_work_day = self.date_utils.get_current_work_day()
        self.load_work_day_data(self.viewing_work_day)

    def load_work_day_data(self, work_day: datetime):
        """
        Load data for a specific work day (uses cache if available)

        Args:
            work_day: The work day to load data for
        """
        print(f"📅 Loading work day: {work_day.date()}")

        # Update UI to show which day we're viewing
        self.update_viewing_date_label()

        # Try to load from cache first
        cached_data = self.daily_cache.load_day_cache(work_day)

        if cached_data:
            print(f"✅ Loaded from cache: {len(cached_data['messages'])} messages")
            self.sentiment_history = cached_data['messages']
            self.breaks_today = cached_data['breaks']
            if self.sentiment_history:
                self.current_sentiment = self.sentiment_history[-1]['sentiment']
        else:
            print(f"⚠️ No cache for {work_day.date()}, filtering from all messages...")
            # Filter messages for this work day from all_messages_cache
            if self.all_messages_cache:
                day_messages = self.date_utils.get_all_messages_for_work_day(
                    self.all_messages_cache, work_day
                )
                print(f"📊 Found {len(day_messages)} messages for this day")

                # Analyze if model is ready
                if self.model_loaded:
                    self.analyze_work_day_messages(day_messages, work_day)
                else:
                    print("⏳ Model not ready - will load data when model loads")
            else:
                print("⚠️ No messages cache available yet")
                self.sentiment_history = []
                self.breaks_today = []

        # Update all UI components
        self.update_all_ui()

    def analyze_work_day_messages(self, messages: list, work_day: datetime):
        """
        Analyze messages for a specific work day and save to cache

        Args:
            messages: List of message dicts for this work day
            work_day: The work day these messages belong to
        """
        self.sentiment_history = []
        self.breaks_today = []

        if not messages:
            print("No messages to analyze")
            return

        print(f"💪 Analyzing {len(messages)} messages for {work_day.date()}...")

        for i, msg in enumerate(messages):
            prev_timestamp = messages[i-1]['timestamp'] if i > 0 else None
            self.analyze_message_text(
                msg['text'],
                msg['timestamp'],
                conversation_file=msg.get('file', ''),
                prev_global_timestamp=prev_timestamp
            )

        # Save to cache
        cache_data = {
            'messages': self.sentiment_history,
            'breaks': self.breaks_today,
            'work_hours': self.calculate_todays_work_hours() if self.sentiment_history else 0,
            'break_count': len(self.breaks_today),
            'message_count': len(self.sentiment_history),
            'avg_sentiment': sum(m['sentiment'] for m in self.sentiment_history) / len(self.sentiment_history) if self.sentiment_history else 0,
        }
        self.daily_cache.save_day_cache(work_day, cache_data)
        print(f"💾 Cached data for {work_day.date()}")

    def update_viewing_date_label(self):
        """Update the date label to show which day we're viewing"""
        current_day = self.date_utils.get_current_work_day()
        date_str = self.date_utils.format_work_day(self.viewing_work_day)

        self.viewing_date_label.setText(date_str)

        # Enable/disable next button based on whether we're at today
        if self.viewing_work_day.date() >= current_day.date():
            self.next_day_btn.setEnabled(False)
        else:
            self.next_day_btn.setEnabled(True)

    def update_all_ui(self):
        """Update all UI components with current data"""
        self.update_mood_display()
        self.update_chart_and_insights()
        self.check_alerts()

    def export_data(self, silent=False):
        """Export sentiment data to JSON"""
        export_dir = self.config.get('export_dir', Path.home() / "ClaudeMood" / "data" / "exports")
        export_dir.mkdir(parents=True, exist_ok=True)

        output_file = export_dir / f"export-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"

        export_data = {
            'export_time': datetime.now().isoformat(),
            'current_sentiment': self.current_sentiment,
            'history': [
                {
                    'timestamp': h['timestamp'].isoformat(),
                    'sentiment': h['sentiment'],
                    'text': h['text']
                }
                for h in self.sentiment_history
            ]
        }

        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)

        if not silent:
            self.live_feed.append(f"✅ Exported to: {output_file}")

        print(f"💾 Exported sentiment data to: {output_file}")

    def detect_break(self, current_time, prev_global_timestamp=None):
        """Detect if there was a break between last message and current message

        Args:
            current_time: Timestamp of current message
            prev_global_timestamp: Timestamp of previous message in GLOBAL timeline (all conversations)
                                  If None, use self.last_message_time (for real-time monitoring)

        Returns:
            Break info dict if break detected, None otherwise
        """
        # Determine the previous message time
        # If prev_global_timestamp is provided, use it (batch analysis with global timeline)
        # Otherwise use self.last_message_time (real-time monitoring)
        prev_time = prev_global_timestamp if prev_global_timestamp is not None else self.last_message_time

        if prev_time is None:
            # First message - no break
            self.last_message_time = current_time
            return None

        health_config = self.config.get('health', {})
        min_break_duration = health_config.get('min_break_duration_minutes', 15)

        time_diff = current_time - prev_time
        minutes_diff = time_diff.total_seconds() / 60

        # Check if this qualifies as a break
        if minutes_diff >= min_break_duration:
            # Only count breaks during the same day
            if prev_time.date() == current_time.date():
                break_info = {
                    'start': prev_time,
                    'end': current_time,
                    'duration_minutes': minutes_diff
                }
                self.breaks_today.append(break_info)
                print(f"☕ Break detected: {minutes_diff:.1f} minutes")
                # Update last_message_time
                self.last_message_time = current_time
                return break_info

        # Always update last_message_time
        self.last_message_time = current_time
        return None

    def calculate_parallel_sessions(self):
        """Calculate current and peak parallel sessions"""
        todays_messages = self.get_todays_messages()

        if not todays_messages:
            return 0, 0

        health_config = self.config.get('health', {})
        session_window_minutes = health_config.get('session_active_window_minutes', 30)
        session_window_seconds = session_window_minutes * 60

        # Track active conversations at each message time
        max_parallel = 0
        current_parallel = 0

        # Get current time
        now = datetime.now()

        # Find conversations active in the last session_window minutes
        recent_cutoff = now - timedelta(seconds=session_window_seconds)
        active_conversations = set()

        for msg in todays_messages:
            if msg['timestamp'] >= recent_cutoff:
                conv_file = msg.get('conversation_file', '')
                if conv_file:
                    active_conversations.add(conv_file)

        current_parallel = len(active_conversations)

        # Calculate peak parallel sessions throughout the day
        # For each message, count how many conversations were active in the preceding window
        for i, msg in enumerate(todays_messages):
            msg_time = msg['timestamp']
            window_start = msg_time - timedelta(seconds=session_window_seconds)

            # Count unique conversations in this window
            active_in_window = set()
            for j in range(i + 1):
                if todays_messages[j]['timestamp'] >= window_start:
                    conv_file = todays_messages[j].get('conversation_file', '')
                    if conv_file:
                        active_in_window.add(conv_file)

            max_parallel = max(max_parallel, len(active_in_window))

        return current_parallel, max_parallel

    def calculate_todays_work_hours(self):
        """Calculate total work hours today (excluding breaks)"""
        todays_messages = self.get_todays_messages()

        if not todays_messages:
            return 0

        # Start = first message, end = last message or now
        start_time = todays_messages[0]['timestamp']
        end_time = todays_messages[-1]['timestamp']

        # If last message was recent (within 30 min), use current time
        health_config = self.config.get('health', {})
        session_window_minutes = health_config.get('session_active_window_minutes', 30)
        if (datetime.now() - end_time).total_seconds() < session_window_minutes * 60:
            end_time = datetime.now()

        total_duration = end_time - start_time
        total_hours = total_duration.total_seconds() / 3600

        # Subtract break time
        total_break_minutes = sum(b['duration_minutes'] for b in self.breaks_today)
        total_break_hours = total_break_minutes / 60

        work_hours = max(0, total_hours - total_break_hours)
        return work_hours

    def calculate_work_health_status(self):
        """Calculate work health status for today"""
        todays_messages = self.get_todays_messages()

        if not todays_messages:
            return {
                'status': 'no_data',
                'label': '⏳ No Data',
                'color': '#999999',
                'warnings': []
            }

        health_config = self.config.get('health', {})

        # Get configuration thresholds
        healthy_min = health_config.get('healthy_min_hours', 4)
        healthy_max = health_config.get('healthy_max_hours', 8)
        overwork_threshold = health_config.get('overwork_threshold', 8)
        extreme_threshold = health_config.get('extreme_threshold', 10)
        late_night_hour = health_config.get('late_night_hour', 22)
        min_breaks = health_config.get('min_breaks_per_day', 2)
        max_gap_hours = health_config.get('max_gap_minutes', 240) / 60  # 4 hours in minutes

        work_hours = self.calculate_todays_work_hours()
        break_count = len(self.breaks_today)
        current_hour = datetime.now().hour
        is_weekend = datetime.now().weekday() >= 5  # Saturday = 5, Sunday = 6

        warnings = []
        status = 'healthy'
        label = '✅ Healthy'
        color = '#4CAF50'

        # Check for extreme overwork
        if work_hours > extreme_threshold:
            status = 'extreme'
            label = '🔴 Extreme Day'
            color = '#D32F2F'
            warnings.append(f"⚠️ Extreme overwork: {work_hours:.1f} hours (limit: {extreme_threshold}h)")

        # Check for overwork
        elif work_hours > overwork_threshold:
            status = 'overwork'
            label = '⚠️ Overwork'
            color = '#FF9800'
            warnings.append(f"⚠️ Overwork detected: {work_hours:.1f} hours (healthy max: {healthy_max}h)")

        # Check if work hours are in healthy range
        elif healthy_min <= work_hours <= healthy_max:
            status = 'healthy'
            label = '✅ Healthy'
            color = '#4CAF50'

        # Check for late night work
        if current_hour >= late_night_hour and todays_messages:
            # Check if there were messages in the last 30 minutes
            last_message_time = todays_messages[-1]['timestamp']
            if (datetime.now() - last_message_time).total_seconds() < 1800:
                warnings.append(f"🌙 Late night work detected (after {late_night_hour}:00)")
                if status == 'healthy':
                    status = 'late_night'
                    label = '🌙 Late Night'
                    color = '#9C27B0'

        # Check for insufficient breaks
        if work_hours > 4 and break_count < min_breaks:
            warnings.append(f"⛔ Insufficient breaks: {break_count} (minimum: {min_breaks})")
            if status == 'healthy':
                status = 'no_breaks'
                label = '⛔ No Breaks'
                color = '#FF5722'

        # Check for weekend work
        if is_weekend:
            warnings.append("📅 Weekend work detected")

        return {
            'status': status,
            'label': label,
            'color': color,
            'work_hours': work_hours,
            'break_count': break_count,
            'warnings': warnings
        }

    def save_daily_snapshot(self):
        """Save daily snapshot of sentiment and work health data"""
        snapshot_config = self.config.get('snapshots', {})

        if not snapshot_config.get('enabled', True):
            return

        snapshots_dir = Path(snapshot_config.get('snapshots_dir', '~/ClaudeMood/data/snapshots')).expanduser()
        snapshots_dir.mkdir(parents=True, exist_ok=True)

        today = datetime.now().date()
        snapshot_file = snapshots_dir / f"{today}.json"

        todays_messages = self.get_todays_messages()
        health_status = self.calculate_work_health_status()

        # Calculate sentiment statistics
        if todays_messages:
            sentiments = [m['sentiment'] for m in todays_messages]
            avg_sentiment = sum(sentiments) / len(sentiments)
            positive_count = len([s for s in sentiments if s > 0.1])
            negative_count = len([s for s in sentiments if s < -0.1])
            neutral_count = len(sentiments) - positive_count - negative_count
        else:
            avg_sentiment = 0
            positive_count = 0
            negative_count = 0
            neutral_count = 0

        snapshot_data = {
            'date': str(today),
            'summary': {
                'total_messages': len(todays_messages),
                'work_hours': health_status.get('work_hours', 0),
                'break_count': health_status.get('break_count', 0),
                'health_status': health_status.get('status', 'no_data'),
                'health_label': health_status.get('label', 'No Data'),
                'sentiment_avg': avg_sentiment,
                'sentiment_positive': positive_count,
                'sentiment_neutral': neutral_count,
                'sentiment_negative': negative_count,
                'warnings': health_status.get('warnings', [])
            },
            'breaks': [
                {
                    'start': b['start'].isoformat(),
                    'end': b['end'].isoformat(),
                    'duration_minutes': b['duration_minutes']
                }
                for b in self.breaks_today
            ],
            'messages': [
                {
                    'timestamp': m['timestamp'].isoformat(),
                    'sentiment': m['sentiment'],
                    'text_preview': m['text'][:100]
                }
                for m in todays_messages
            ]
        }

        with open(snapshot_file, 'w', encoding='utf-8') as f:
            json.dump(snapshot_data, f, indent=2, ensure_ascii=False)

        print(f"💾 Daily snapshot saved: {snapshot_file}")

    def closeEvent(self, event):
        """Cleanup on close"""
        print("👋 Closing ClaudeMood...")

        # Save daily snapshot before closing
        if self.sentiment_history:
            self.save_daily_snapshot()

        # Stop conversation monitor
        if hasattr(self, 'observer'):
            self.observer.stop()
            self.observer.join()

        # Stop app file monitor (if auto-restart was enabled)
        if hasattr(self, 'app_observer'):
            self.app_observer.stop()
            self.app_observer.join()

        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look

    window = ClaudeMoodApp()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
