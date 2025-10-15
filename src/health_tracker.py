"""
Work Health Tracking for ClaudeMood
Tracks breaks, work hours, and health status
"""
import json
from datetime import datetime, timedelta
from pathlib import Path


class HealthTracker:
    """Tracks work health metrics like breaks, work hours, and health status"""

    def __init__(self, app):
        """Initialize health tracker with app instance

        Args:
            app: The ClaudeMoodApp instance
        """
        self.app = app

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
        prev_time = prev_global_timestamp if prev_global_timestamp is not None else self.app.last_message_time

        if prev_time is None:
            # First message - no break
            self.app.last_message_time = current_time
            return None

        health_config = self.app.config.get('health', {})
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
                self.app.breaks_today.append(break_info)
                print(f"☕ Break detected: {minutes_diff:.1f} minutes")
                # Update last_message_time
                self.app.last_message_time = current_time
                return break_info

        # Always update last_message_time
        self.app.last_message_time = current_time
        return None

    def calculate_parallel_sessions(self):
        """Calculate current and peak parallel sessions"""
        todays_messages = self.app.get_todays_messages()

        if not todays_messages:
            return 0, 0

        health_config = self.app.config.get('health', {})
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
        todays_messages = self.app.get_todays_messages()

        if not todays_messages:
            return 0

        # Start = first message, end = last message or now
        start_time = todays_messages[0]['timestamp']
        end_time = todays_messages[-1]['timestamp']

        # If last message was recent (within 30 min), use current time
        health_config = self.app.config.get('health', {})
        session_window_minutes = health_config.get('session_active_window_minutes', 30)
        if (datetime.now() - end_time).total_seconds() < session_window_minutes * 60:
            end_time = datetime.now()

        total_duration = end_time - start_time
        total_hours = total_duration.total_seconds() / 3600

        # Subtract break time
        total_break_minutes = sum(b['duration_minutes'] for b in self.app.breaks_today)
        total_break_hours = total_break_minutes / 60

        work_hours = max(0, total_hours - total_break_hours)
        return work_hours

    def calculate_work_health_status(self):
        """Calculate work health status for today"""
        todays_messages = self.app.get_todays_messages()

        if not todays_messages:
            return {
                'status': 'no_data',
                'label': '⏳ No Data',
                'color': '#999999',
                'warnings': []
            }

        health_config = self.app.config.get('health', {})

        # Get configuration thresholds
        healthy_min = health_config.get('healthy_min_hours', 4)
        healthy_max = health_config.get('healthy_max_hours', 8)
        overwork_threshold = health_config.get('overwork_threshold', 8)
        extreme_threshold = health_config.get('extreme_threshold', 10)
        late_night_hour = health_config.get('late_night_hour', 22)
        min_breaks = health_config.get('min_breaks_per_day', 2)
        max_gap_hours = health_config.get('max_gap_minutes', 240) / 60  # 4 hours in minutes

        work_hours = self.calculate_todays_work_hours()
        break_count = len(self.app.breaks_today)
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
        snapshot_config = self.app.config.get('snapshots', {})

        if not snapshot_config.get('enabled', True):
            return

        snapshots_dir = Path(snapshot_config.get('snapshots_dir', '~/ClaudeMood/data/snapshots')).expanduser()
        snapshots_dir.mkdir(parents=True, exist_ok=True)

        today = datetime.now().date()
        snapshot_file = snapshots_dir / f"{today}.json"

        todays_messages = self.app.get_todays_messages()
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
                for b in self.app.breaks_today
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
