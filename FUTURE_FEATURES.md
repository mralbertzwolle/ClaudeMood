# ClaudeMood - Future Features Plan

**Based on:** SparkBuddy conversation-analytics scripts
**Created:** 2025-10-14
**Status:** Planning Phase

---

## 🎯 Vision

Transform ClaudeMood from a real-time sentiment tracker into a comprehensive work health and productivity analytics platform.

---

## 📊 Planned Features (Priority Order)

### 1. **Work Health Monitoring** 🏥
**Priority:** HIGH
**Inspired by:** `analyze_work_health.py`

**Features:**
- ✅ Healthy day detection (4-8h work, 2+ breaks, ends before 22:00)
- ⚠️ Overwork day alerts (>8 hours)
- 🔴 Extreme day warnings (>10 hours)
- 🌙 Late night detection (working after 22:00)
- ⛔ No-break warnings (>4h without break)
- 📅 Weekend work tracking

**Implementation:**
- Add health category calculation to daily stats
- Show health status in Chart & Insights tab
- Color-coded daily indicators
- Weekly health summary

---

### 2. **Break Pattern Analysis** ☕
**Priority:** HIGH
**Inspired by:** Work health analysis

**Features:**
- Detect breaks (>15 min gaps between messages)
- Track breaks per day
- Break quality metrics (duration, timing)
- Break recommendations
- "Time for a break!" reminders

**Implementation:**
- Add break detection to message processing
- New "Breaks" section in Statistics tab
- Visual timeline showing work/break periods
- Desktop notifications for break reminders

---

### 3. **Historical Data Snapshots** 💾
**Priority:** HIGH
**Inspired by:** Monthly snapshot system

**Features:**
- Daily data snapshots (JSON format)
- Monthly reports (Markdown format)
- Historical trend analysis
- Data preservation before deletion
- Export/import functionality

**Implementation:**
- Add snapshot generation to export_data()
- Create `data/snapshots/` directory
- Automated monthly report generation
- Historical data viewer in UI

---

### 4. **Detailed Daily Breakdown** 📋
**Priority:** MEDIUM
**Inspired by:** `analyze_detailed_daily.py`

**Features:**
- Day-by-day table view
- Columns: Date, Hours, Sessions, Breaks, Messages, Start/End times, Status
- Weekly summaries
- Day-of-week patterns (Mon-Sun averages)
- Health status flags per day

**Implementation:**
- New "Daily Log" tab
- Table widget with sortable columns
- Export to CSV functionality
- Filterable by date range

---

### 5. **Session Duration & Gaps** ⏱️
**Priority:** MEDIUM
**Inspired by:** Realistic time analysis

**Features:**
- Detect conversation sessions (gap threshold: 30 min)
- Session start/end times
- Active time vs. total time
- Parallel conversation detection
- Parallelism factor metric

**Implementation:**
- Add session detection algorithm
- Show sessions in timeline view
- Session statistics in Stats tab
- "Current session" indicator

---

### 6. **Monthly & Yearly Projections** 📈
**Priority:** MEDIUM
**Inspired by:** Multi-project analysis

**Features:**
- Monthly hours projection
- Yearly hours estimate
- Equivalent work weeks calculation
- Productivity trends over time
- Comparison with previous months

**Implementation:**
- Add projection calculations
- New "Projections" section
- Trend charts (monthly comparison)
- Goal setting and tracking

---

### 7. **Multi-Day Sentiment Trends** 📉
**Priority:** MEDIUM
**Current:** Only shows today's data
**Future:** Historical sentiment analysis

**Features:**
- Week-over-week sentiment comparison
- Monthly sentiment averages
- Identify stress patterns
- Correlation with work hours
- Sentiment by day of week

**Implementation:**
- Extend chart to show multiple days
- Add date range selector
- Sentiment heatmap (calendar view)
- Export sentiment reports

---

### 8. **Productivity Metrics** 🚀
**Priority:** LOW
**Inspired by:** Messages per hour analysis

**Features:**
- Messages per hour rate
- Peak productivity hours
- Focus time detection (long continuous sessions)
- Productivity by project/conversation
- Comparison with personal averages

**Implementation:**
- Add productivity calculations
- New "Productivity" tab
- Hourly heatmap visualization
- Peak hours identification

---

### 9. **Work-Life Balance Score** ⚖️
**Priority:** LOW
**Inspired by:** Health recommendations

**Features:**
- Overall balance score (0-100)
- Factors: hours, breaks, late nights, weekends
- Weekly balance report
- Recommendations for improvement
- Streak tracking (healthy days in a row)

**Implementation:**
- Balance score algorithm
- Score widget on main screen
- Historical score tracking
- Achievement badges

---

### 10. **Automated Recommendations** 💡
**Priority:** LOW
**Inspired by:** Health recommendations system

**Features:**
- Smart suggestions based on patterns
- "Take a break" notifications
- "Stop for today" alerts (after 8h)
- Late night warnings (22:00 reminder)
- Weekend work discouragement

**Implementation:**
- Recommendation engine
- Desktop notifications
- Settings for notification thresholds
- Snooze/dismiss functionality

---

## 🔧 Technical Requirements

### Dependencies to Add:
- **matplotlib** ✅ (already added for charts)
- **numpy** ✅ (already added for trends)
- **pandas** (optional, for advanced data analysis)

### Data Structure Changes:
- Add session tracking to sentiment_history
- Store break information
- Add health status per day
- Extended timestamp metadata

### Configuration Additions:
```json
{
  "health": {
    "healthy_min_hours": 4,
    "healthy_max_hours": 8,
    "overwork_threshold": 8,
    "extreme_threshold": 10,
    "late_night_hour": 22,
    "min_break_duration_minutes": 15,
    "max_gap_minutes": 30,
    "min_breaks_per_day": 2
  },
  "notifications": {
    "enabled": true,
    "break_reminders": true,
    "late_night_warnings": true,
    "daily_limit_alerts": true
  },
  "snapshots": {
    "enabled": true,
    "daily_auto_save": true,
    "monthly_reports": true
  }
}
```

---

## 📅 Implementation Roadmap

### Phase 1: Core Health Features (Week 1-2)
- Work health monitoring
- Break detection
- Late night warnings
- Historical snapshots

### Phase 2: Data & Insights (Week 3-4)
- Daily breakdown table
- Session detection
- Monthly projections
- Multi-day sentiment trends

### Phase 3: Advanced Analytics (Week 5-6)
- Productivity metrics
- Work-life balance score
- Automated recommendations
- Achievement system

### Phase 4: Polish & Export (Week 7-8)
- CSV/PDF export
- Report generation
- Data import/restore
- Settings UI improvements

---

## 🎨 UI/UX Improvements

### New Tabs to Add:
1. **Daily Log** - Detailed day-by-day breakdown
2. **Health** - Work health metrics and recommendations
3. **Productivity** - Messages/hour, peak times, focus sessions
4. **History** - Multi-day/week/month view with comparisons

### Current Tab Updates:
- **Chart & Insights** - Extend to multi-day view
- **Statistics** - Add projections and comparisons
- **Alerts** - Add health-based alerts (breaks, late night, etc.)

---

## 📊 Data Preservation Strategy

**Problem:** No auto-deletion in ClaudeMood (unlike Claude Code), but still need historical analysis.

**Solution:**
- Daily JSON snapshots in `data/snapshots/YYYY-MM-DD.json`
- Monthly markdown reports in `data/reports/YYYY-MM.md`
- Automatic compression of old data (>90 days)
- Export/import for backup

**Format:**
```json
{
  "date": "2025-10-14",
  "summary": {
    "total_hours": 8.5,
    "messages": 156,
    "sentiment_avg": 0.23,
    "health_status": "healthy",
    "breaks": 3,
    "late_night": false
  },
  "sessions": [...],
  "messages": [...]
}
```

---

## 🔗 Integration with SparkBuddy Analytics

**Potential Cross-Tool Analysis:**
- Import SparkBuddy snapshot data
- Combined work hours (coding + Claude conversations)
- Project-specific sentiment correlation
- Unified health dashboard

---

## 💭 Future Ideas (Brainstorm)

- **AI Insights:** GPT-based analysis of work patterns
- **Team Mode:** Compare anonymized metrics with team
- **Voice Reminders:** "Time for a break!" audio alerts
- **Mobile App:** iOS/Android companion for notifications
- **Browser Extension:** Track all Claude conversations (web + desktop)
- **Integration:** Slack/Discord notifications
- **Gamification:** Achievements for healthy work patterns
- **Coach Mode:** Weekly review sessions with recommendations

---

## 📝 Notes

- Keep privacy as priority (100% local)
- All features optional (configurable)
- Maintain lightweight performance
- Focus on actionable insights
- Don't become overwhelming

---

**Last Updated:** 2025-10-14
**Next Review:** 2025-11-01
