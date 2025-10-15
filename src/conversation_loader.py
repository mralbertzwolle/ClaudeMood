"""
Conversation Loader - Handles loading and parsing conversations
"""
import json
from pathlib import Path
from datetime import datetime


class ConversationLoader:
    """Loads conversations from various formats"""

    def __init__(self, conversations_dir, date_utils):
        self.conversations_dir = conversations_dir
        self.date_utils = date_utils

    def load_all_messages(self, max_files=100):
        """
        Load all messages from conversation files

        Returns:
            list: All messages with timestamps sorted chronologically
        """
        all_messages = []

        # Check format type
        if self.conversations_dir.name == "projects":
            all_messages = self._load_jsonl_messages(max_files)
        else:
            all_messages = self._load_json_messages(max_files)

        # Sort by timestamp
        all_messages.sort(key=lambda m: m['timestamp'])
        return all_messages

    def _load_jsonl_messages(self, max_files):
        """Load messages from JSONL files (Claude Code projects format)"""
        all_messages = []
        jsonl_files = list(self.conversations_dir.rglob("*.jsonl"))

        if not jsonl_files:
            return all_messages

        # Sort by modification time
        jsonl_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        for conv_file in jsonl_files[:max_files]:
            try:
                with open(conv_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            message_obj = data.get('message', data)

                            if message_obj.get('role') == 'user':
                                content = message_obj.get('content', '')
                                timestamp_str = data.get('timestamp', '')

                                # Extract text from content
                                text = self._extract_text_from_content(content)

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

        return all_messages

    def _load_json_messages(self, max_files):
        """Load messages from JSON files (old conversations format)"""
        all_messages = []
        json_files = list(self.conversations_dir.glob("*.json"))

        if not json_files:
            return all_messages

        # Sort by modification time
        json_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        for conv_file in json_files[:max_files]:
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

        return all_messages

    def _extract_text_from_content(self, content):
        """Extract text from content (handles list or string format)"""
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get('type') == 'text':
                    text_parts.append(part.get('text', ''))
            return ' '.join(text_parts)
        else:
            return content

    def parse_single_conversation(self, file_path):
        """
        Parse a single conversation file and return the last user message

        Returns:
            tuple: (text, timestamp) or (None, None) if no valid message found
        """
        file_path_obj = Path(file_path)

        # Handle JSONL format
        if file_path_obj.suffix == '.jsonl':
            return self._parse_jsonl_conversation(file_path_obj)
        else:
            return self._parse_json_conversation(file_path_obj)

    def _parse_jsonl_conversation(self, file_path):
        """Parse JSONL conversation and return last user message"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                last_user_message = None
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        message_obj = data.get('message', data)

                        if message_obj.get('role') == 'user':
                            content = message_obj.get('content', '')
                            last_user_message = self._extract_text_from_content(content)
                    except (json.JSONDecodeError, KeyError):
                        continue

                if last_user_message and len(last_user_message) >= 10:
                    return last_user_message, datetime.now()
        except Exception as e:
            print(f"❌ Error parsing {file_path}: {e}")

        return None, None

    def _parse_json_conversation(self, file_path):
        """Parse JSON conversation and return last user message"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                messages = data.get('chat', [])

                if not messages:
                    return None, None

                # Get last message
                last_msg = messages[-1]
                text = last_msg.get('text', '')
                sender = last_msg.get('sender', 'unknown')

                if sender == 'human' and text and len(text) >= 10:
                    return text, datetime.now()
        except Exception as e:
            print(f"❌ Error parsing {file_path}: {e}")

        return None, None
