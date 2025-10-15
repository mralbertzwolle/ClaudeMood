"""
Data Exporter - Handles exporting sentiment data
"""
import json
from datetime import datetime
from pathlib import Path


class DataExporter:
    """Export sentiment data to various formats"""

    def __init__(self, config):
        self.config = config
        self.export_dir = config.get('export_dir', Path.home() / "ClaudeMood" / "data" / "exports")
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def export_sentiment_history(self, sentiment_history, current_sentiment, silent=False):
        """Export sentiment data to JSON"""
        output_file = self.export_dir / f"export-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"

        export_data = {
            'export_time': datetime.now().isoformat(),
            'current_sentiment': current_sentiment,
            'history': [
                {
                    'timestamp': h['timestamp'].isoformat(),
                    'sentiment': h['sentiment'],
                    'text': h['text']
                }
                for h in sentiment_history
            ]
        }

        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)

        if not silent:
            print(f"💾 Data exported: {output_file}")

        return output_file
