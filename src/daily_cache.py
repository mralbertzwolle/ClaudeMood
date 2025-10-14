#!/usr/bin/env python3
"""
Daily Cache System for ClaudeMood
Stores analyzed sentiment data per day to avoid re-analyzing everything
"""
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class DailyCache:
    """Manages daily cache files for sentiment analysis results"""

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_file(self, date: datetime) -> Path:
        """Get cache file path for a specific date"""
        date_str = date.strftime('%Y-%m-%d')
        return self.cache_dir / f"cache_{date_str}.json"

    def load_day_cache(self, date: datetime) -> Optional[Dict]:
        """Load cache for a specific day"""
        cache_file = self.get_cache_file(date)

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Convert timestamp strings back to datetime
                for msg in data.get('messages', []):
                    msg['timestamp'] = datetime.fromisoformat(msg['timestamp'])
                for brk in data.get('breaks', []):
                    brk['start'] = datetime.fromisoformat(brk['start'])
                    brk['end'] = datetime.fromisoformat(brk['end'])
                return data
        except Exception as e:
            print(f"⚠️ Error loading cache {cache_file}: {e}")
            return None

    def save_day_cache(self, date: datetime, data: Dict):
        """Save cache for a specific day"""
        cache_file = self.get_cache_file(date)

        # Convert datetime objects to ISO strings for JSON
        save_data = {
            'date': date.strftime('%Y-%m-%d'),
            'cached_at': datetime.now().isoformat(),
            'messages': [],
            'breaks': [],
            'work_hours': data.get('work_hours', 0),
            'break_count': data.get('break_count', 0),
            'message_count': data.get('message_count', 0),
            'avg_sentiment': data.get('avg_sentiment', 0),
        }

        # Convert messages
        for msg in data.get('messages', []):
            save_data['messages'].append({
                'timestamp': msg['timestamp'].isoformat(),
                'text': msg['text'],
                'sentiment': msg['sentiment'],
                'conversation_file': msg.get('conversation_file', ''),
            })

        # Convert breaks
        for brk in data.get('breaks', []):
            save_data['breaks'].append({
                'start': brk['start'].isoformat(),
                'end': brk['end'].isoformat(),
                'duration_minutes': brk['duration_minutes'],
            })

        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved cache for {date.strftime('%Y-%m-%d')} ({len(save_data['messages'])} messages)")
        except Exception as e:
            print(f"⚠️ Error saving cache {cache_file}: {e}")

    def get_cached_message_ids(self, date: datetime) -> set:
        """Get set of message IDs (timestamp+text hash) that are already cached"""
        cache = self.load_day_cache(date)
        if not cache:
            return set()

        cached_ids = set()
        for msg in cache.get('messages', []):
            # Use timestamp + first 50 chars of text as unique ID
            msg_id = f"{msg['timestamp']}_{msg['text'][:50]}"
            cached_ids.add(msg_id)

        return cached_ids

    def get_available_dates(self) -> List[datetime]:
        """Get list of dates that have cache files"""
        cache_files = list(self.cache_dir.glob("cache_*.json"))
        dates = []

        for cache_file in cache_files:
            try:
                # Extract date from filename: cache_2025-10-14.json
                date_str = cache_file.stem.replace('cache_', '')
                date = datetime.strptime(date_str, '%Y-%m-%d')
                dates.append(date)
            except:
                continue

        return sorted(dates, reverse=True)  # Most recent first

    def cleanup_old_caches(self, keep_days: int = 30):
        """Delete cache files older than keep_days"""
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        cache_files = list(self.cache_dir.glob("cache_*.json"))

        deleted_count = 0
        for cache_file in cache_files:
            try:
                date_str = cache_file.stem.replace('cache_', '')
                date = datetime.strptime(date_str, '%Y-%m-%d')

                if date < cutoff_date:
                    cache_file.unlink()
                    deleted_count += 1
            except:
                continue

        if deleted_count > 0:
            print(f"🗑️ Cleaned up {deleted_count} old cache files")
