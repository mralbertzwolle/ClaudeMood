"""
UI Updater for ClaudeMood
Handles all UI update operations
"""
from datetime import datetime
import matplotlib.dates as mdates
import numpy as np


class UIUpdater:
    """Handles all UI update operations for the ClaudeMood app"""

    def __init__(self, app):
        """
        Initialize UI updater

        Args:
            app: Reference to the main ClaudeMoodApp instance
        """
        self.app = app

    def get_todays_messages(self):
        """Get all messages from current work day (respects day_start_hour config)"""
        # Use viewing_work_day instead of calendar day
        # This ensures we filter messages correctly when day_start_hour != 0
        return [h for h in self.app.sentiment_history
                if self.app.date_utils.get_work_day_for_timestamp(h['timestamp']) == self.app.viewing_work_day]

    def update_mood_display(self):
        """Update the mood display based on current sentiment"""
        sentiment = self.app.current_sentiment
        thresholds = self.app.config.get('mood_thresholds', {})

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

        self.app.mood_emoji.setText(emoji)
        self.app.mood_label.setText(f"Current Mood: {mood} ({sentiment:+.2f})")
        self.app.mood_label.setStyleSheet(f"color: {color};")

        # Update progress bar
        bar_value = int(sentiment * 100)
        self.app.mood_bar.setValue(bar_value)

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

        self.app.mood_bar.setStyleSheet(f"""
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
            self.app.chart_widget.update_chart(timestamps, sentiments, self.app.breaks_today)

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
            self.app.time_info_label.setText(info_text)
        else:
            # Fast path: Skip expensive chart rendering for empty data
            # Just update the info label - chart will show cached "No data" message
            self.app.time_info_label.setText("⏳ No messages for this day. Select another day or start coding!")

    def update_dashboard(self):
        """Update dashboard metrics"""
        todays_messages = self.get_todays_messages()
        health_status = self.app.health_tracker.calculate_work_health_status()

        # Update metric cards
        work_hours = health_status.get('work_hours', 0)
        break_count = health_status.get('break_count', 0)

        self.app.dashboard_work_hours.setText(f"{work_hours:.1f}h")
        self.app.dashboard_breaks.setText(str(break_count))
        self.app.dashboard_messages.setText(str(len(todays_messages)))

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

        self.app.dashboard_health_emoji.setText(status_emoji)
        self.app.dashboard_health_status.setText(status_label.split()[1] if ' ' in status_label else status_label)
        self.app.dashboard_health_status.setStyleSheet(f"color: {status_color};")

        # Update dashboard summary
        if not todays_messages:
            summary_text = "⏳ No work session data yet today. Start coding to see your metrics!"
        else:
            start_time = todays_messages[0]['timestamp']
            avg_sentiment = sum(m['sentiment'] for m in todays_messages) / len(todays_messages)

            # Calculate total break time
            total_break_minutes = sum(b['duration_minutes'] for b in self.app.breaks_today) if self.app.breaks_today else 0

            # Calculate parallel sessions
            current_parallel, peak_parallel = self.app.health_tracker.calculate_parallel_sessions()

            warnings = health_status.get('warnings', [])

            summary_text = f"""
<b>📅 Today's Summary</b><br><br>

<b>Session Started:</b> {start_time.strftime('%H:%M')}<br>
<b>Net Work Time:</b> {work_hours:.1f} hours<br>
<b>Total Break Time:</b> {total_break_minutes:.0f} minutes ({len(self.app.breaks_today)} breaks)<br>
<b>Average Sentiment:</b> {avg_sentiment:+.2f} {'😊' if avg_sentiment > 0.1 else '😐' if avg_sentiment > -0.1 else '😞'}<br>
<b>Active Sessions:</b> {current_parallel} currently | Peak: {peak_parallel} parallel<br><br>
"""

            if warnings:
                summary_text += "<b>⚠️ Health Alerts:</b><br>"
                for warning in warnings[:3]:  # Show max 3 warnings
                    summary_text += f"• {warning}<br>"
            else:
                summary_text += "<b>✅ All health metrics are good!</b>"

        self.app.dashboard_summary.setText(summary_text)

    def refresh_ui(self):
        """Refresh UI elements"""
        # Only update if data is loaded
        if not self.app.data_loaded or not self.app.sentiment_history:
            return

        self.update_dashboard()
        self.update_statistics()
        self.update_health_metrics()
        self.check_alerts()
        self.update_chart_and_insights()

    def update_statistics(self):
        """Update statistics display"""
        if not self.app.sentiment_history:
            return

        total = len(self.app.sentiment_history)
        positive = len([h for h in self.app.sentiment_history if h['sentiment'] > 0.1])
        negative = len([h for h in self.app.sentiment_history if h['sentiment'] < -0.1])
        neutral = total - positive - negative

        avg_sentiment = sum(h['sentiment'] for h in self.app.sentiment_history) / total

        # Phase 1: Add break and work hour statistics
        todays_messages = self.get_todays_messages()
        work_hours = self.app.health_tracker.calculate_todays_work_hours() if todays_messages else 0
        break_count = len(self.app.breaks_today)

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

Current Mood Score: {self.app.current_sentiment:+.3f}

⏰ Work & Break Statistics (Today):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Work Hours: {work_hours:.1f}h (net)
Breaks Taken: {break_count}
"""

        if self.app.breaks_today:
            total_break_time = sum(b['duration_minutes'] for b in self.app.breaks_today)
            avg_break = total_break_time / len(self.app.breaks_today)
            stats_text += f"Total Break Time: {total_break_time:.0f} min ({total_break_time/60:.1f}h)\n"
            stats_text += f"Average Break: {avg_break:.0f} min\n"

        self.app.stats_label.setText(stats_text)

    def update_health_metrics(self):
        """Update health metrics display (Phase 1)"""
        health_status = self.app.health_tracker.calculate_work_health_status()
        todays_messages = self.get_todays_messages()

        if not todays_messages:
            self.app.health_label.setText("⏳ No work session data yet today.")
            return

        work_hours = health_status.get('work_hours', 0)
        break_count = health_status.get('break_count', 0)
        status_label = health_status.get('label', 'No Data')
        status_color = health_status.get('color', '#999999')
        warnings = health_status.get('warnings', [])

        # Calculate break statistics
        if self.app.breaks_today:
            avg_break_duration = sum(b['duration_minutes'] for b in self.app.breaks_today) / len(self.app.breaks_today)
            total_break_time = sum(b['duration_minutes'] for b in self.app.breaks_today)
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
        if self.app.breaks_today:
            health_text += "\nBreak Schedule:\n"
            for i, brk in enumerate(self.app.breaks_today, 1):
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
        health_config = self.app.config.get('health', {})
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

        self.app.health_label.setText(health_text)

    def check_alerts(self):
        """Check for mood and health alerts (Phase 1)"""
        if len(self.app.sentiment_history) < 5:
            return

        alert_config = self.app.config.get('alerts', {})
        high_stress = alert_config.get('high_stress_threshold', -0.3)
        very_negative = alert_config.get('very_negative_threshold', -0.5)
        consecutive_count = alert_config.get('consecutive_negative_count', 4)

        recent = self.app.sentiment_history[-5:]
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

        if self.app.current_sentiment < very_negative:
            mood_alerts.append("🚨 Very negative mood")

        # Phase 1: Health-based alerts
        notifications_config = self.app.config.get('notifications', {})
        if notifications_config.get('enabled', True):
            health_status = self.app.health_tracker.calculate_work_health_status()
            todays_messages = self.get_todays_messages()

            if todays_messages:
                work_hours = health_status.get('work_hours', 0)
                break_count = health_status.get('break_count', 0)

                # Late night warning
                if notifications_config.get('late_night_warnings', True):
                    current_hour = datetime.now().hour
                    late_night_hour = self.app.config.get('health', {}).get('late_night_hour', 22)
                    if current_hour >= late_night_hour:
                        last_message_time = todays_messages[-1]['timestamp']
                        if (datetime.now() - last_message_time).total_seconds() < 1800:
                            health_alerts.append(f"🌙 Late night ({late_night_hour}:00+)")

                # Daily limit alert
                if notifications_config.get('daily_limit_alerts', True):
                    overwork_threshold = self.app.config.get('health', {}).get('overwork_threshold', 8)
                    extreme_threshold = self.app.config.get('health', {}).get('extreme_threshold', 10)

                    if work_hours > extreme_threshold:
                        health_alerts.append(f"🔴 {work_hours:.1f}h worked (extreme!)")
                    elif work_hours > overwork_threshold:
                        health_alerts.append(f"⚠️ {work_hours:.1f}h worked")

                # Break reminder
                if notifications_config.get('break_reminders', True):
                    min_breaks = self.app.config.get('health', {}).get('min_breaks_per_day', 2)
                    if work_hours > 4 and break_count < min_breaks:
                        health_alerts.append(f"☕ {break_count} breaks only")

                    # Check time since last break
                    if self.app.last_message_time:
                        time_since_last = (datetime.now() - self.app.last_message_time).total_seconds() / 60
                        max_gap_minutes = self.app.config.get('health', {}).get('max_gap_minutes', 240)
                        if time_since_last > max_gap_minutes and time_since_last < max_gap_minutes + 30:
                            health_alerts.append(f"⏰ {time_since_last/60:.1f}h straight")

        # Combine alerts
        if mood_alerts:
            alerts.append("😞 MOOD ALERTS:\n━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" + "\n".join(mood_alerts))

        if health_alerts:
            alerts.append("\n🏥 HEALTH ALERTS:\n━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" + "\n".join(health_alerts))

        if alerts:
            # Update alerts tab
            self.app.alerts_label.setText("\n".join(alerts))
            self.app.alerts_label.setStyleSheet("color: #F44336; font-weight: bold;")

            # Show subtle banner (NO BLINKING!)
            all_alerts = mood_alerts + health_alerts
            banner_text = " • ".join(all_alerts)  # Simple separator
            self.app.alert_banner.setText(banner_text)
            self.app.alert_banner.setStyleSheet("""
                background-color: #fff5f5;
                color: #e74c3c;
                padding: 10px;
                border-radius: 6px;
                border-left: 4px solid #e74c3c;
            """)
            self.app.alert_banner.show()
        else:
            # No alerts - hide banner
            self.app.alert_banner.hide()

            self.app.alerts_label.setText("✅ All good! No alerts.\n\n💪 Keep coding with healthy habits!")
            self.app.alerts_label.setStyleSheet("color: #4CAF50;")
