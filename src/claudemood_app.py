#!/usr/bin/env python3
"""
ClaudeMood - Real-time Developer Sentiment Tracker
Monitors Claude Code conversations and tracks your mood while coding
"""
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from daily_cache import DailyCache
from date_utils import DateUtils
from widgets import DirectorySelectionDialog
from workers import ModelLoader, DataLoader, AnalysisWorker, SearchWorker
from config_utils import load_config, save_config
from file_monitors import AppFileMonitor, ConversationMonitor
from ui_builder import build_ui
from health_tracker import HealthTracker
from ui_updater import UIUpdater
from conversation_loader import ConversationLoader
from message_analyzer import MessageAnalyzer
from data_exporter import DataExporter
from directory_finder import DirectoryFinder

from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt6.QtCore import QTimer, QProcess, pyqtSignal
from watchdog.observers import Observer


class ClaudeMoodApp(QMainWindow):
    # Qt signal for thread-safe file change notifications
    conversation_changed_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        # Config
        self.config = load_config()
        ui_config = self.config.get('ui', {})
        self.setWindowTitle("ClaudeMood - Developer Sentiment Tracker 🎭")
        self.setGeometry(100, 100, ui_config.get('window_width', 900), ui_config.get('window_height', 700))

        # Data
        self.sentiment_history = []
        self.current_sentiment = 0.0
        self.daily_stats = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0})
        self.breaks_today = []
        self.last_message_time = None
        self.last_reset_date = datetime.now().date()
        self.sentiment_analyzer = None
        self.model_loaded = False
        self.data_loaded = False
        self.all_messages_cache = []

        # Helpers
        cache_dir = Path.home() / "ClaudeMood" / "cache"
        self.daily_cache = DailyCache(cache_dir)
        day_start_hour = self.config.get('analysis', {}).get('day_start_hour', 4)
        self.date_utils = DateUtils(day_start_hour)
        self.viewing_work_day = self.date_utils.get_current_work_day()

        # Services
        self.health_tracker = HealthTracker(self)
        self.ui_updater = UIUpdater(self)
        self.message_analyzer = MessageAnalyzer(self)
        self.data_exporter = DataExporter(self.config)
        self.conversation_loader = ConversationLoader(self.config.get('conversations_dir'), self.date_utils)

        # Setup
        self.conversation_changed_signal.connect(self.handle_conversation_changed)
        self.init_ui()
        self.setup_monitoring()
        if self.config.get('auto_restart', {}).get('enabled', True):
            self.setup_auto_restart()

        # Timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.refresh_ui)
        self.update_timer.start(ui_config.get('update_interval_seconds', 5) * 1000)

        # Load
        self.load_model_async()
        self.load_data_async()

    def load_model_async(self):
        """Load sentiment model"""
        model_cache_dir = self.config.get('model_cache_dir', Path.home() / "ClaudeMood" / "models" / "hf_cache")
        self.model_loader = ModelLoader(model_cache_dir)
        self.model_loader.finished.connect(self.on_model_loaded)
        self.model_loader.error.connect(self.on_model_error)
        self.model_loader.start()
        self.mood_label.setText("Loading model...")
        self.mood_emoji.setText("⏳")

    def on_model_loaded(self, sentiment_pipeline):
        """Model loaded"""
        self.sentiment_analyzer = sentiment_pipeline
        self.model_loaded = True
        self.mood_label.setText("Model loaded - Waiting for data...")
        self.mood_emoji.setText("✅")
        if self.data_loaded:
            self.reanalyze_all_messages()

    def on_model_error(self, error_msg):
        """Model error"""
        self.mood_label.setText(f"Error: {error_msg}")
        self.mood_emoji.setText("❌")

    def load_data_async(self):
        """Load conversation data"""
        self.data_loader = DataLoader(self)
        self.data_loader.progress.connect(self.on_data_progress)
        self.data_loader.finished.connect(self.on_data_loaded)
        self.data_loader.start()

    def on_data_progress(self, message):
        """Data loading progress"""
        if hasattr(self, 'dashboard_summary'):
            self.dashboard_summary.setText(f"⏳ {message}")

    def on_data_loaded(self):
        """Data loaded"""
        self.data_loaded = True
        self.refresh_ui()
        if hasattr(self, 'mood_label'):
            self.update_mood_display()

    def init_ui(self):
        """Initialize the user interface"""
        build_ui(self)

    def setup_monitoring(self):
        conv_dir = DirectoryFinder(self, self.config).find_and_select_directory(self.config.get('conversations_dir'))
        if conv_dir:
            self.monitor = ConversationMonitor(self.on_conversation_changed, conv_dir)
            self.observer = Observer()
            self.observer.schedule(self.monitor, str(conv_dir), recursive=True)
            self.observer.start()

    def setup_auto_restart(self):
        watch = self.config.get('auto_restart', {}).get('watch_files', [])
        if watch:
            root = Path(__file__).parent.parent
            self.app_monitor = AppFileMonitor(self.on_source_file_changed)
            self.app_observer = Observer()
            self.app_observer.schedule(self.app_monitor, str(root / "src"), recursive=False)
            self.app_observer.schedule(self.app_monitor, str(root), recursive=False)
            self.app_observer.start()

    def on_source_file_changed(self, fp):
        if QMessageBox.question(self, "Restart Required", f"{Path(fp).name} changed\n\nRestart?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            self.restart_app()

    def on_conversation_changed(self, fp): self.conversation_changed_signal.emit(fp)
    def handle_conversation_changed(self, fp): self.analyze_conversation(fp)
    def reanalyze_all_messages(self): self.load_recent_conversations(); self.refresh_ui()
    def restart_app(self):
        if self.sentiment_history: self.export_data(silent=True)
        QProcess.startDetached(sys.executable, sys.argv); QApplication.quit()

    def load_recent_conversations(self):
        """Load conversations"""
        today = datetime.now()
        cached = self.daily_cache.load_day_cache(today)
        if cached:
            self.sentiment_history, self.breaks_today = cached['messages'], cached['breaks']
            if self.sentiment_history: self.current_sentiment = self.sentiment_history[-1]['sentiment']
        else:
            self.sentiment_history, self.breaks_today = [], []

        self.all_messages_cache = self.conversation_loader.load_all_messages(max_files=100)
        today_msgs = self.date_utils.get_all_messages_for_work_day(
            self.all_messages_cache, self.date_utils.get_work_day_for_timestamp(today))

        if self.sentiment_analyzer:
            cached_ids = self.daily_cache.get_cached_message_ids(today)
            new_msgs = [m for m in today_msgs if f"{m['timestamp']}_{m['text'][:50]}" not in cached_ids]
            for msg in new_msgs:
                prev = self.sentiment_history[-1]['timestamp'] if self.sentiment_history else None
                self.message_analyzer.analyze_and_add_to_history(msg['text'], msg['timestamp'], msg.get('file', ''), prev)

        self.save_cache(today)
        self.update_viewing_date_label()

    def check_daily_reset(self, ts):
        if ts.date() > self.last_reset_date:
            if self.sentiment_history: self.health_tracker.save_daily_snapshot()
            self.breaks_today, self.last_message_time = [], None
            self.last_reset_date = ts.date()

    # Analysis delegations
    def analyze_single_message(self, text, ts, cf='', prev=None):
        return self.message_analyzer.analyze_single_message(text, ts, cf, prev)
    def get_mood_emoji(self, sentiment): return self.message_analyzer.get_mood_emoji(sentiment)
    def update_live_feed(self): self.message_analyzer.update_live_feed()

    def analyze_conversation(self, file_path):
        """Analyze conversation file"""
        if self.sentiment_analyzer:
            text, timestamp = self.conversation_loader.parse_single_conversation(file_path)
            if text:
                self.message_analyzer.analyze_and_add_to_history(text, timestamp, str(file_path))

    def save_cache(self, work_day):
        """Save cache"""
        avg = sum(m['sentiment'] for m in self.sentiment_history) / len(self.sentiment_history) if self.sentiment_history else 0
        self.daily_cache.save_day_cache(work_day, {
            'messages': self.sentiment_history, 'breaks': self.breaks_today,
            'work_hours': self.health_tracker.calculate_todays_work_hours() if self.sentiment_history else 0,
            'break_count': len(self.breaks_today), 'message_count': len(self.sentiment_history), 'avg_sentiment': avg
        })

    # UI delegations
    def update_mood_display(self): self.ui_updater.update_mood_display()
    def update_chart_and_insights(self): self.ui_updater.update_chart_and_insights()
    def update_dashboard(self): self.ui_updater.update_dashboard()
    def refresh_ui(self): self.ui_updater.refresh_ui()
    def update_statistics(self): self.ui_updater.update_statistics()
    def update_health_metrics(self): self.ui_updater.update_health_metrics()
    def check_alerts(self): self.ui_updater.check_alerts()

    # Date navigation
    def go_to_previous_day(self):
        self.viewing_work_day = self.date_utils.get_previous_work_day(self.viewing_work_day)
        self.load_work_day_data(self.viewing_work_day)
    def go_to_next_day(self):
        new_day = self.date_utils.get_next_work_day(self.viewing_work_day)
        if new_day <= self.date_utils.get_current_work_day():
            self.viewing_work_day = new_day
            self.load_work_day_data(self.viewing_work_day)
    def go_to_today(self):
        self.viewing_work_day = self.date_utils.get_current_work_day()
        self.load_work_day_data(self.viewing_work_day)

    def load_work_day_data(self, work_day: datetime):
        """Load work day"""
        self.update_viewing_date_label()
        cached = self.daily_cache.load_day_cache(work_day)
        if cached:
            self.sentiment_history = cached['messages']
            self.breaks_today = cached['breaks']
            if self.sentiment_history:
                self.current_sentiment = self.sentiment_history[-1]['sentiment']
            return
        if self.all_messages_cache:
            day_msgs = self.date_utils.get_all_messages_for_work_day(self.all_messages_cache, work_day)
            if self.model_loaded and day_msgs:
                self.start_background_analysis(day_msgs, work_day)
            else:
                self.sentiment_history, self.breaks_today = [], []
                self.update_all_ui()
        else:
            self.sentiment_history, self.breaks_today = [], []
            self.update_all_ui()

    def start_background_analysis(self, messages: list, work_day: datetime):
        """Start analysis"""
        self.analysis_progress_label.setText(f"💪 {work_day.strftime('%Y-%m-%d')} ({len(messages)} msgs)")
        self.analysis_progress_bar.setMaximum(len(messages))
        self.analysis_progress_bar.setValue(0)
        self.analysis_progress_bar.show()
        self.analysis_worker = AnalysisWorker(self, messages, work_day)
        self.analysis_worker.progress.connect(lambda c, t: self.analysis_progress_bar.setValue(c))
        self.analysis_worker.finished.connect(lambda d: self.on_analysis_complete(d, work_day))
        self.analysis_worker.error.connect(lambda e: [self.analysis_progress_label.setText(f"❌ {e}"), self.analysis_progress_bar.hide()])
        self.analysis_worker.start()

    def on_analysis_complete(self, cache_data: dict, work_day: datetime):
        """Complete"""
        self.sentiment_history = cache_data['messages']
        self.breaks_today = cache_data['breaks']
        if self.sentiment_history:
            self.current_sentiment = self.sentiment_history[-1]['sentiment']
        self.daily_cache.save_day_cache(work_day, cache_data)
        self.analysis_progress_bar.hide()
        self.analysis_progress_label.setText("✅ Complete!")
        self.update_all_ui()


    def update_viewing_date_label(self):
        self.viewing_date_label.setText(self.date_utils.format_work_day(self.viewing_work_day))
        self.next_day_btn.setEnabled(self.viewing_work_day.date() < self.date_utils.get_current_work_day().date())

    def update_all_ui(self):
        self.update_mood_display(); self.update_chart_and_insights(); self.check_alerts()

    def export_data(self, silent=False):
        return self.data_exporter.export_sentiment_history(self.sentiment_history, self.current_sentiment, silent)

    def closeEvent(self, event):
        """Cleanup"""
        if self.sentiment_history:
            self.health_tracker.save_daily_snapshot()
        if hasattr(self, 'observer'):
            self.observer.stop()
            self.observer.join()
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
