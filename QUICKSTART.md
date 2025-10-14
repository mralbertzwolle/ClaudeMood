# 🚀 ClaudeMood Quick Start Guide

## Installation ✅ (Already Done!)

Your ClaudeMood installation is complete and ready to use!

## 🎯 Launch ClaudeMood

```bash
cd ~/ClaudeMood
./launch.sh
```

The app will:
1. Load configuration from `config.json`
2. Initialize RobBERT model (~2-3 seconds)
3. Open the dashboard
4. Start monitoring your conversations

## ⚙️ First-Time Configuration

On first run, `config.json` is auto-created from `config.json.example`.

### Quick Tweaks

Edit `~/ClaudeMood/config.json`:

```bash
# Use your favorite editor
code ~/ClaudeMood/config.json
# or
nano ~/ClaudeMood/config.json
```

### Common Adjustments

```json
{
  // Make alert more/less sensitive
  "alerts": {
    "high_stress_threshold": -0.2,  // Default: -0.3 (lower = more sensitive)
    "very_negative_threshold": -0.4  // Default: -0.5
  },

  // Change window size
  "ui": {
    "window_width": 1200,  // Default: 900
    "window_height": 800   // Default: 700
  },

  // Disable auto-restart during normal use
  "auto_restart": {
    "enabled": false  // Default: true
  }
}
```

**Tip**: When `auto_restart` is enabled, the app asks to restart when you save `config.json`!

## 📊 Dashboard Overview

### Main Display
- **Big Emoji** - Your current mood (😄 🙂 😐 😕 😞)
- **Mood Label** - Text mood + score
- **Progress Bar** - Visual sentiment scale

### Tabs

1. **Live Feed** - Recent messages with sentiment scores
2. **Statistics** - Overall breakdown (positive/neutral/negative %)
3. **Alerts** - Stress warnings and recommendations

### Buttons

- **Refresh Now** - Manually reload conversations
- **Export Data** - Save sentiment history to JSON
- **Quit** - Close app (stops all monitoring)

## 🎭 Understanding Sentiment Scores

| Score | Emoji | Meaning | Color |
|-------|-------|---------|-------|
| > +0.3 | 😄 | Happy - Crushing it! | Green |
| +0.1 to +0.3 | 🙂 | Good - Productive flow | Light Green |
| -0.1 to +0.1 | 😐 | Neutral - Normal coding | Yellow |
| -0.3 to -0.1 | 😕 | Stressed - Frustration | Orange |
| < -0.3 | 😞 | Frustrated - Break time! | Red |

## ⚠️ Alert System

### High Stress (-0.3 average)
**Trigger**: Recent 5 messages average below -0.3
**Action**: Consider taking a 5-10 minute break

### Multiple Negatives (4+ in a row)
**Trigger**: 4 or more negative messages consecutively
**Action**: Step away from keyboard, stretch, hydrate

### Very Negative (< -0.5)
**Trigger**: Current mood drops below -0.5
**Action**: Stop coding, take a proper break (15-30 min)

## 💾 Export & Data

### Manual Export
Click **"Export Data"** → saves to `~/ClaudeMood/data/exports/`

### Auto-Export (on restart)
When auto-restart triggers, sentiment history is automatically saved

### Export Format
```json
{
  "export_time": "2025-10-14T17:30:00",
  "current_sentiment": -0.125,
  "history": [
    {
      "timestamp": "2025-10-14T14:30:00",
      "sentiment": 0.45,
      "text": "Perfect! Dat werkt goed"
    }
  ]
}
```

## 🔄 Auto-Restart Feature

### When Enabled
- Monitors `src/*.py` and `config.json`
- Shows dialog on file change
- Auto-exports before restart
- Great for development/tweaking

### When Disabled
- No file monitoring overhead
- Recommended for daily use
- Manually restart if needed

## 🛠️ Advanced Usage

### Custom Directories

```json
{
  "conversations_dir": "/custom/path/to/conversations",
  "export_dir": "/custom/export/location"
}
```

### Different Alert Thresholds

```json
{
  "alerts": {
    "high_stress_threshold": -0.2,  // More sensitive
    "consecutive_negative_count": 3  // Fewer negatives trigger
  }
}
```

### Longer Rolling Average

```json
{
  "analysis": {
    "rolling_average_window": 20  // Default: 10
  }
}
```

Longer window = smoother, less reactive mood changes

## 📈 Interpreting Trends

### Daily Patterns
- **Morning**: Often neutral/slightly negative (cold start)
- **Afternoon**: Usually highest positivity (peak energy)
- **Evening**: Drops off (fatigue sets in)

### Weekly Patterns
- **Monday**: Lower scores (weekend recovery)
- **Mid-week**: Peak productivity
- **Friday**: Lower intensity (winding down)

### Watch For
- **Consistent negativity** - Burnout risk
- **Extreme swings** - Stress/frustration cycles
- **Flat neutral** - Possible disengagement

## 🐛 Troubleshooting

### App Won't Start
```bash
# Check Python version
/tmp/robbert-venv/bin/python --version
# Should be 3.11.x

# Test PyQt6
/tmp/robbert-venv/bin/python -c "from PyQt6.QtWidgets import QApplication; print('OK')"
```

### No Conversations Detected
```bash
# Check conversations directory
ls ~/Library/Application\ Support/Claude/conversations/
```

### Model Loading Issues
```bash
# Check model cache
ls ~/ClaudeMood/models/hf_cache/
# Should contain RobBERT model (~1.7GB)
```

## 💡 Pro Tips

1. **Baseline calibration** - Run for 1 week, then adjust thresholds to your personal baseline
2. **Break reminders** - Use alerts as mandatory break triggers
3. **Weekly exports** - Export data every Friday for trend analysis
4. **Disable auto-restart** - Turn off when not actively developing
5. **Window positioning** - Keep dashboard visible in corner while coding

## 📞 Support

For issues or questions:
1. Check `README.md` for detailed docs
2. Review `CHANGELOG.md` for recent changes
3. Inspect `config.json` for misconfigurations

---

**Happy coding with healthy moods! 🎭💻**
