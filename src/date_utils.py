#!/usr/bin/env python3
"""
Date Utilities for ClaudeMood
Handles custom day boundaries (e.g., 04:00-04:00 instead of 00:00-24:00)
"""
from datetime import datetime, timedelta
from typing import Tuple


class DateUtils:
    """Utilities for handling custom day boundaries"""

    def __init__(self, day_start_hour: int = 4):
        """
        Initialize with custom day start hour

        Args:
            day_start_hour: Hour when the "day" starts (default: 4 = 04:00)
        """
        self.day_start_hour = day_start_hour

    def get_work_day_for_timestamp(self, timestamp: datetime) -> datetime:
        """
        Get the work day date for a given timestamp

        For example, with day_start_hour=4:
        - 2025-01-10 02:00 belongs to work day 2025-01-09 (still working late!)
        - 2025-01-10 05:00 belongs to work day 2025-01-10 (new day started at 04:00)

        Args:
            timestamp: The timestamp to check

        Returns:
            datetime: The work day date (always at 00:00 for consistency)
        """
        # If before day_start_hour, it belongs to previous calendar day
        if timestamp.hour < self.day_start_hour:
            work_day = (timestamp - timedelta(days=1)).date()
        else:
            work_day = timestamp.date()

        # Return as datetime at 00:00 for consistency
        return datetime.combine(work_day, datetime.min.time())

    def get_work_day_boundaries(self, work_day: datetime) -> Tuple[datetime, datetime]:
        """
        Get start and end times for a work day

        For example, with day_start_hour=4:
        Work day 2025-01-10:
        - Start: 2025-01-10 04:00:00
        - End:   2025-01-11 04:00:00 (exclusive)

        Args:
            work_day: The work day (datetime at 00:00)

        Returns:
            Tuple[datetime, datetime]: (start_time, end_time)
        """
        start_time = datetime.combine(work_day.date(), datetime.min.time())
        start_time = start_time.replace(hour=self.day_start_hour)
        end_time = start_time + timedelta(days=1)

        return start_time, end_time

    def is_in_work_day(self, timestamp: datetime, work_day: datetime) -> bool:
        """
        Check if a timestamp belongs to a specific work day

        Args:
            timestamp: The timestamp to check
            work_day: The work day to check against

        Returns:
            bool: True if timestamp is in this work day
        """
        start_time, end_time = self.get_work_day_boundaries(work_day)
        return start_time <= timestamp < end_time

    def get_current_work_day(self) -> datetime:
        """Get the current work day based on current time"""
        return self.get_work_day_for_timestamp(datetime.now())

    def get_previous_work_day(self, work_day: datetime) -> datetime:
        """Get the previous work day"""
        return work_day - timedelta(days=1)

    def get_next_work_day(self, work_day: datetime) -> datetime:
        """Get the next work day"""
        return work_day + timedelta(days=1)

    def format_work_day(self, work_day: datetime) -> str:
        """Format work day for display"""
        today = self.get_current_work_day()
        yesterday = self.get_previous_work_day(today)

        if work_day.date() == today.date():
            return f"Today ({work_day.strftime('%Y-%m-%d')})"
        elif work_day.date() == yesterday.date():
            return f"Yesterday ({work_day.strftime('%Y-%m-%d')})"
        else:
            return work_day.strftime('%Y-%m-%d')

    def get_all_messages_for_work_day(self, all_messages: list, work_day: datetime) -> list:
        """
        Filter messages that belong to a specific work day

        Args:
            all_messages: List of message dicts with 'timestamp' key
            work_day: The work day to filter for

        Returns:
            list: Messages belonging to this work day
        """
        start_time, end_time = self.get_work_day_boundaries(work_day)
        return [
            msg for msg in all_messages
            if start_time <= msg['timestamp'] < end_time
        ]
