#!/usr/bin/env python3
"""
File system monitors for ClaudeMood
"""
from datetime import datetime
from pathlib import Path
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
        self.conversations_dir = Path(conversations_dir)
        self.last_modified = {}

    def on_modified(self, event):
        """Handle file modification events"""
        if event.is_directory:
            return

        # Only watch JSONL files (Claude Code format)
        if not event.src_path.endswith('.jsonl'):
            return

        # Must be in conversations directory
        event_path = Path(event.src_path)
        if not str(event_path).startswith(str(self.conversations_dir)):
            return

        # Debounce: ignore rapid successive changes
        now = datetime.now()
        if event.src_path in self.last_modified:
            if (now - self.last_modified[event.src_path]).total_seconds() < 2:
                return

        self.last_modified[event.src_path] = now
        print(f"📝 Conversation changed: {event.src_path}")

        # Notify callback
        self.callback(event.src_path)
