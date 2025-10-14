# ClaudeMood Changelog

## [v1.1.0] - 2025-10-14

### ✨ New Features

#### Configuration System
- **config.json** - Complete configuration file system
  - All paths configurable (conversations, data, exports, model cache)
  - UI settings (window size, update interval)
  - Analysis settings (rolling window, text length)
  - Alert thresholds customizable
  - Mood thresholds adjustable
- **config.json.example** - Template for easy setup
- **Auto-generated config** - Created from template on first run
- **Gitignored** - Personal settings stay private

#### Auto-Restart System
- **File monitoring** - Watches source files and config for changes
- **Smart debouncing** - Ignores rapid successive changes (2s cooldown)
- **User confirmation** - Dialog before restart
- **Auto-export** - Saves sentiment history before restart
- **Configurable** - Enable/disable via `config.json`
- **Multi-file watch** - Monitors `src/*.py` and `config.json`

### 🔧 Improvements

#### Flexibility
- **Configurable directories** - All paths now from config
- **Adjustable thresholds** - Fine-tune alert and mood boundaries
- **Window preferences** - Save your preferred size
- **Update frequency** - Control refresh rate

#### Developer Experience
- **Silent exports** - Export without UI notification (for auto-save)
- **Better cleanup** - Properly stops all file observers
- **Path expansion** - Automatic `~` expansion in config paths

### 📋 Configuration Options

```json
{
  "conversations_dir": "~/Library/Application Support/Claude/conversations",
  "data_dir": "~/ClaudeMood/data",
  "export_dir": "~/ClaudeMood/data/exports",
  "model_cache_dir": "~/ClaudeMood/models/hf_cache",

  "ui": {
    "window_width": 900,
    "window_height": 700,
    "update_interval_seconds": 5
  },

  "analysis": {
    "rolling_average_window": 10,
    "max_text_length": 512
  },

  "alerts": {
    "high_stress_threshold": -0.3,
    "very_negative_threshold": -0.5,
    "consecutive_negative_count": 4
  },

  "mood_thresholds": {
    "very_positive": 0.3,
    "positive": 0.1,
    "neutral_high": 0.1,
    "neutral_low": -0.1,
    "negative": -0.3
  },

  "auto_restart": {
    "enabled": true,
    "watch_files": [
      "src/claudemood_app.py",
      "src/sentiment_analyzer.py",
      "config.json"
    ]
  }
}
```

### 🔒 Privacy & Git

- **.gitignore** - Comprehensive ignore rules
  - `config.json` (personal settings)
  - `data/` (sentiment history)
  - `models/` (large ML models)
  - Python bytecode, OS files, IDE files

### 📚 Documentation

- **README updated** - Complete configuration guide
- **Project structure** - Clear file organization
- **Usage tips** - Best practices for configuration

---

## [v1.0.0] - 2025-10-14

### 🎉 Initial Release

#### Core Features
- Real-time conversation monitoring
- RobBERT Dutch sentiment analysis
- PyQt6 desktop interface
- Live mood display with emoji
- Progress bar (-100 to +100)
- Tabbed interface (Live Feed, Statistics, Alerts)
- Smart stress alerts
- Data export to JSON

#### AI Analysis
- RobBERT model integration (93% accuracy)
- Context-aware sentiment detection
- Rolling average calculation
- Positive/Neutral/Negative classification

#### UI/UX
- Modern Fusion style
- Real-time updates
- Color-coded mood states
- Statistics dashboard
- Alert notifications
- Manual refresh button

#### Data Management
- Conversation file monitoring
- Sentiment history tracking
- JSON export functionality
- Timestamp tracking
