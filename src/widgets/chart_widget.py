"""
Sentiment Chart Widget for ClaudeMood
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


class SentimentChartWidget(QWidget):
    """Widget for displaying beautiful sentiment and work intensity charts"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Create matplotlib figure with 2 subplots
        self.figure = Figure(figsize=(10, 7), dpi=100, facecolor='#f8f9fa')
        self.canvas = FigureCanvas(self.figure)

        # Create layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Initialize empty charts
        self.ax1 = None  # Work intensity chart
        self.ax2 = None  # Sentiment chart
        self.update_chart([], [], [])

    def update_chart(self, timestamps, sentiments, breaks_today):
        """Update both sentiment and work intensity charts"""
        self.figure.clear()

        if not timestamps or len(timestamps) < 2:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No data yet...\nStart coding to see your activity!',
                    ha='center', va='center', fontsize=16, color='#6c757d',
                    weight='bold')
            ax.set_facecolor('#f8f9fa')
            ax.axis('off')
            self.canvas.draw()
            return

        # Create 2 subplots
        self.ax1 = self.figure.add_subplot(211)  # Work intensity (top)
        self.ax2 = self.figure.add_subplot(212)  # Sentiment (bottom)

        # === WORK INTENSITY CHART (TOP) ===
        self._plot_work_intensity(timestamps)

        # === SENTIMENT CHART (BOTTOM) ===
        self._plot_sentiment(timestamps, sentiments, breaks_today)

        # Tight layout
        self.figure.tight_layout(pad=2.0)
        self.canvas.draw()

    def _plot_work_intensity(self, timestamps):
        """Plot work intensity (messages per hour)"""
        # Calculate messages per 30-minute bucket
        bucket_counts = defaultdict(int)

        for ts in timestamps:
            # Round to 30-minute bucket
            bucket = ts.replace(minute=(ts.minute // 30) * 30, second=0, microsecond=0)
            bucket_counts[bucket] += 1

        # Sort buckets
        buckets = sorted(bucket_counts.keys())
        counts = [bucket_counts[b] for b in buckets]

        if not buckets:
            return

        # Plot as bar chart with gradient
        bars = self.ax1.bar(buckets, counts, width=0.02, color='#667eea',
                            alpha=0.8, edgecolor='none')

        # Add gradient effect to bars
        for bar, count in zip(bars, counts):
            # Color intensity based on count
            intensity = min(count / max(counts), 1.0)
            bar.set_color(plt.cm.RdYlGn(0.3 + intensity * 0.5))

        # Styling
        self.ax1.set_facecolor('#ffffff')
        self.ax1.set_title('💪 Work Intensity (Messages per 30min)',
                          fontsize=14, fontweight='bold', color='#2c3e50', pad=15)
        self.ax1.set_ylabel('Messages', fontsize=11, color='#495057', fontweight='600')
        self.ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
        self.ax1.spines['top'].set_visible(False)
        self.ax1.spines['right'].set_visible(False)
        self.ax1.spines['left'].set_color('#dee2e6')
        self.ax1.spines['bottom'].set_color('#dee2e6')

        # Format x-axis
        self.ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        self.figure.autofmt_xdate(rotation=45)

        # Add max indicator
        max_count = max(counts)
        max_idx = counts.index(max_count)
        self.ax1.annotate(f'Peak: {max_count} msgs',
                         xy=(buckets[max_idx], max_count),
                         xytext=(10, 10), textcoords='offset points',
                         bbox=dict(boxstyle='round,pad=0.5', fc='#ffeaa7', alpha=0.8),
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='#fd79a8'),
                         fontsize=9, fontweight='bold')

    def _plot_sentiment(self, timestamps, sentiments, breaks_today):
        """Plot sentiment over time with beautiful gradients"""
        # Plot sentiment line with gradient fill
        x_numeric = mdates.date2num(timestamps)

        # Smooth the line with interpolation (if scipy available)
        try:
            if len(timestamps) > 10:
                from scipy.interpolate import make_interp_spline

                # Remove duplicates (scipy can't handle duplicate x values)
                unique_indices = []
                seen_times = set()
                for i, x in enumerate(x_numeric):
                    if x not in seen_times:
                        seen_times.add(x)
                        unique_indices.append(i)

                if len(unique_indices) > 3:  # Need at least 4 points for cubic spline
                    x_unique = [x_numeric[i] for i in unique_indices]
                    y_unique = [sentiments[i] for i in unique_indices]

                    x_smooth = np.linspace(x_unique[0], x_unique[-1], 300)
                    spl = make_interp_spline(x_unique, y_unique, k=3)
                    sentiments_smooth = spl(x_smooth)
                    timestamps_smooth = mdates.num2date(x_smooth)
                else:
                    # Not enough unique points for interpolation
                    timestamps_smooth = timestamps
                    sentiments_smooth = sentiments
            else:
                timestamps_smooth = timestamps
                sentiments_smooth = sentiments
        except (ImportError, ValueError) as e:
            # scipy not available or interpolation failed - use original data
            print(f"⚠️ Interpolation skipped: {e}")
            timestamps_smooth = timestamps
            sentiments_smooth = sentiments

        # Plot line with shadow effect
        self.ax2.plot(timestamps_smooth, sentiments_smooth, color='#5f27cd',
                     linewidth=3, label='Sentiment', zorder=5, alpha=0.9)

        # Fill area under curve with gradient
        positive_mask = np.array(sentiments_smooth) > 0
        negative_mask = np.array(sentiments_smooth) <= 0

        # Positive gradient (green)
        self.ax2.fill_between(timestamps_smooth, 0, sentiments_smooth,
                             where=positive_mask,
                             interpolate=True, alpha=0.3,
                             color='#00b894', label='Positive mood')

        # Negative gradient (red)
        self.ax2.fill_between(timestamps_smooth, 0, sentiments_smooth,
                             where=negative_mask,
                             interpolate=True, alpha=0.3,
                             color='#ff7675', label='Negative mood')

        # Visualize breaks as red zones
        if breaks_today:
            for brk in breaks_today:
                self.ax2.axvspan(brk['start'], brk['end'],
                                alpha=0.15, color='#d63031', zorder=1)
                # Add break label
                mid_time = brk['start'] + (brk['end'] - brk['start']) / 2
                self.ax2.text(mid_time, 0.9, f"☕ {brk['duration_minutes']:.0f}m",
                             ha='center', va='center', fontsize=8,
                             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8),
                             color='#d63031', fontweight='bold', zorder=10)

        # Add zero line
        self.ax2.axhline(y=0, color='#b2bec3', linestyle='--', linewidth=1.5, alpha=0.6, zorder=2)

        # Add trend line
        if len(x_numeric) >= 3:
            z = np.polyfit(x_numeric, sentiments, 1)
            p = np.poly1d(z)
            trend_line = p(x_numeric)
            trend_color = '#00b894' if z[0] > 0 else '#ff7675'
            self.ax2.plot(timestamps, trend_line, '--', color=trend_color,
                        linewidth=2, alpha=0.7, label=f'Trend {"↑" if z[0] > 0 else "↓"}', zorder=3)

        # Styling
        self.ax2.set_facecolor('#ffffff')
        self.ax2.set_title('😊 Sentiment Over Time',
                          fontsize=14, fontweight='bold', color='#2c3e50', pad=15)
        self.ax2.set_xlabel('Time', fontsize=11, color='#495057', fontweight='600')
        self.ax2.set_ylabel('Sentiment', fontsize=11, color='#495057', fontweight='600')
        self.ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.8, zorder=0)
        self.ax2.legend(loc='upper left', fontsize=9, framealpha=0.9, edgecolor='#dee2e6')
        self.ax2.set_ylim(-1.1, 1.1)

        # Remove top and right spines
        self.ax2.spines['top'].set_visible(False)
        self.ax2.spines['right'].set_visible(False)
        self.ax2.spines['left'].set_color('#dee2e6')
        self.ax2.spines['bottom'].set_color('#dee2e6')

        # Format x-axis
        self.ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        self.figure.autofmt_xdate(rotation=45)
