"""
File system monitors for ClaudeMood
"""
from pathlib import Path
from datetime import datetime
from watchdog.events import FileSystemEventHandler


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
