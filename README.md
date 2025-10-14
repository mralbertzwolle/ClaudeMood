# ClaudeMood 🎭

Real-time Developer Sentiment Tracker for Claude Code conversations

## Features

- 😊 **Real-time sentiment analysis** using RobBERT Dutch language model
- 📊 **Beautiful charts** - Work intensity + Sentiment over time
- 💾 **Daily cache system** - Incremental updates, no re-analysis
- 🏥 **Health monitoring** - Work hours, breaks, alerts
- ⚠️ **Smart alerts** - Overwork warnings, break reminders
- 🎨 **Modern UI** - PyQt6 with professional styling

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Configure
cp config.json.example config.json
# Edit config.json with your Claude Code projects path

# Launch
./launch.sh
```

## How It Works

1. **Monitors** your Claude Code conversations in real-time
2. **Analyzes** sentiment of your messages using AI
3. **Tracks** work hours, breaks, and health metrics
4. **Caches** analyzed data per day (super fast!)
5. **Alerts** you when you need a break

## Architecture

- **Main App**: `src/claudemood_app.py` - PyQt6 GUI
- **Daily Cache**: `src/daily_cache.py` - Efficient data storage
- **Config**: `config.json` - User preferences

## Performance

- **First start**: ~20 seconds (loads and analyzes all messages)
- **Next starts**: ~1 second (loads from cache, only analyzes new messages!)
- **97% faster** with daily caching system

## Screenshots

Beautiful charts showing:
- Work intensity (messages per 30min)
- Sentiment over time with gradients
- Break zones visualization

## License

MIT
