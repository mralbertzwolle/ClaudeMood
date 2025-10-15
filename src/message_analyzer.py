"""
Message Analyzer - Handles sentiment analysis of messages
"""


class MessageAnalyzer:
    """Analyzes sentiment of text messages"""

    def __init__(self, app):
        self.app = app
        self.config = app.config

    def analyze_single_message(self, text, timestamp, conversation_file='', prev_global_timestamp=None):
        """Analyze a single message and return the result (for background analysis)"""
        try:
            if not text or len(text) < 10:
                return None

            if not self.app.sentiment_analyzer:
                return None

            # Detect break
            break_duration = None
            if prev_global_timestamp:
                gap = (timestamp - prev_global_timestamp).total_seconds() / 60
                min_break = self.config.get('health', {}).get('min_break_duration_minutes', 15)
                if gap >= min_break:
                    break_duration = gap

            # Analyze sentiment
            sentiment = self._analyze_text_sentiment(text)

            # Build result
            message_data = {
                'timestamp': timestamp,
                'sentiment': sentiment,
                'text': text[:100],
                'break_before': break_duration is not None,
                'conversation_file': conversation_file
            }

            result = {
                'message_data': message_data,
                'break_detected': break_duration is not None,
                'break_data': None
            }

            if break_duration is not None:
                result['break_data'] = {
                    'start': prev_global_timestamp,
                    'end': timestamp,
                    'duration_minutes': break_duration
                }

            return result

        except Exception as e:
            print(f"❌ Error analyzing single message: {e}")
            return None

    def analyze_and_add_to_history(self, text, timestamp, conversation_file='', prev_global_timestamp=None):
        """Analyze sentiment and add to history"""
        try:
            if not text or len(text) < 10:
                return

            if not self.app.sentiment_analyzer:
                return

            # Check for daily reset
            self.app.check_daily_reset(timestamp)

            # Detect break
            break_info = self.app.health_tracker.detect_break(timestamp, prev_global_timestamp=prev_global_timestamp)

            # Analyze sentiment
            sentiment = self._analyze_text_sentiment(text)

            # Add to history
            self.app.sentiment_history.append({
                'timestamp': timestamp,
                'sentiment': sentiment,
                'text': text[:100],
                'break_before': break_info is not None,
                'conversation_file': conversation_file
            })

            # Update current sentiment
            window_size = self.config.get('analysis', {}).get('rolling_average_window', 10)
            recent = [h['sentiment'] for h in self.app.sentiment_history[-window_size:]]
            self.app.current_sentiment = sum(recent) / len(recent)

            # Update UI
            self.update_live_feed()
            self.app.update_mood_display()

        except Exception as e:
            print(f"❌ Error analyzing message: {e}")

    def _analyze_text_sentiment(self, text):
        """Analyze sentiment using the loaded model"""
        max_length = self.config.get('analysis', {}).get('max_text_length', 512)
        result = self.app.sentiment_analyzer(text[:max_length])[0]
        label = result['label'].lower()
        score = result['score']

        if label == 'positive':
            return score
        elif label == 'negative':
            return -score
        else:
            return 0.0

    def update_live_feed(self):
        """Update Live Feed with last 25 messages"""
        recent = self.app.sentiment_history[-25:]
        self.app.live_feed.clear()

        for entry in recent:
            mood_emoji = self.get_mood_emoji(entry['sentiment'])
            break_indicator = " ☕" if entry.get('break_before', False) else ""
            self.app.live_feed.append(
                f"[{entry['timestamp'].strftime('%H:%M')}] {mood_emoji} "
                f"{entry['sentiment']:+.2f}{break_indicator}: {entry['text'][:80]}"
            )

    def get_mood_emoji(self, sentiment):
        """Get mood emoji for a sentiment value"""
        thresholds = self.config.get('mood_thresholds', {})
        if sentiment > thresholds.get('very_positive', 0.3):
            return "😄"
        elif sentiment > thresholds.get('positive', 0.1):
            return "🙂"
        elif sentiment > thresholds.get('neutral_low', -0.1):
            return "😐"
        elif sentiment > thresholds.get('negative', -0.3):
            return "😕"
        else:
            return "😞"
