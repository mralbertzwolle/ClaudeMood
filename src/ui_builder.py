"""
UI Builder for ClaudeMood
Builds the complete user interface
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QTextEdit, QTabWidget, QScrollArea
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from widgets import SentimentChartWidget


def build_ui(app):
    """Build the complete UI for the ClaudeMood application

    Args:
        app: The ClaudeMoodApp instance
    """
    # Apply modern stylesheet
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f5f7fa;
        }
        QWidget {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
        }
        QLabel {
            color: #2c3e50;
        }
        QPushButton {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 600;
            min-width: 100px;
        }
        QPushButton:hover {
            background-color: #2980b9;
        }
        QPushButton:pressed {
            background-color: #21618c;
        }
        QTextEdit {
            background-color: white;
            border: 1px solid #dfe6e9;
            border-radius: 8px;
            padding: 12px;
            font-size: 12px;
            font-family: 'SF Mono', 'Monaco', 'Courier New', monospace;
            color: #2c3e50;
        }
        QTabWidget::pane {
            border: 1px solid #dfe6e9;
            border-radius: 8px;
            background-color: white;
            padding: 10px;
        }
        QTabBar::tab {
            background-color: #ecf0f1;
            color: #7f8c8d;
            padding: 10px 20px;
            margin-right: 2px;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            font-weight: 600;
            font-size: 13px;
        }
        QTabBar::tab:selected {
            background-color: white;
            color: #3498db;
        }
        QTabBar::tab:hover {
            background-color: #e8ecef;
        }
        QProgressBar {
            border: 1px solid #dfe6e9;
            border-radius: 8px;
            text-align: center;
            height: 30px;
            background-color: #ecf0f1;
            font-weight: 600;
            font-size: 13px;
        }
        QProgressBar::chunk {
            border-radius: 7px;
        }
    """)

    central_widget = QWidget()
    app.setCentralWidget(central_widget)

    layout = QVBoxLayout()
    layout.setSpacing(15)
    layout.setContentsMargins(20, 20, 20, 20)
    central_widget.setLayout(layout)

    # === HEADER ===
    header = QLabel("ClaudeMood")
    header.setFont(QFont("SF Pro Display", 26, QFont.Weight.Bold))
    header.setAlignment(Qt.AlignmentFlag.AlignCenter)
    header.setStyleSheet("""
        color: #2c3e50;
        padding: 15px;
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #667eea, stop:1 #764ba2);
        -webkit-background-clip: text;
        border-radius: 10px;
    """)
    layout.addWidget(header)

    # === DATE NAVIGATION BAR ===
    date_nav_widget = QWidget()
    date_nav_widget.setStyleSheet("""
        QWidget {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 10px;
        }
    """)
    date_nav_layout = QHBoxLayout()
    date_nav_layout.setContentsMargins(15, 10, 15, 10)
    date_nav_widget.setLayout(date_nav_layout)

    # Previous day button
    app.prev_day_btn = QPushButton("← Previous Day")
    app.prev_day_btn.setFont(QFont("SF Pro Text", 12))
    app.prev_day_btn.setStyleSheet("""
        QPushButton {
            background-color: #667eea;
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 6px;
            font-weight: 600;
        }
        QPushButton:hover {
            background-color: #5568d3;
        }
        QPushButton:pressed {
            background-color: #4451b8;
        }
    """)
    app.prev_day_btn.clicked.connect(app.go_to_previous_day)
    date_nav_layout.addWidget(app.prev_day_btn)

    # Current date display
    app.viewing_date_label = QLabel("Today")
    app.viewing_date_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    app.viewing_date_label.setFont(QFont("SF Pro Text", 14, QFont.Weight.Bold))
    app.viewing_date_label.setStyleSheet("color: #2c3e50; padding: 5px;")
    date_nav_layout.addWidget(app.viewing_date_label, 1)  # stretch factor 1

    # Next day button
    app.next_day_btn = QPushButton("Next Day →")
    app.next_day_btn.setFont(QFont("SF Pro Text", 12))
    app.next_day_btn.setStyleSheet("""
        QPushButton {
            background-color: #667eea;
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 6px;
            font-weight: 600;
        }
        QPushButton:hover {
            background-color: #5568d3;
        }
        QPushButton:pressed {
            background-color: #4451b8;
        }
        QPushButton:disabled {
            background-color: #cbd5e0;
            color: #a0aec0;
        }
    """)
    app.next_day_btn.clicked.connect(app.go_to_next_day)
    date_nav_layout.addWidget(app.next_day_btn)

    # "Today" quick button
    app.today_btn = QPushButton("Today")
    app.today_btn.setFont(QFont("SF Pro Text", 12))
    app.today_btn.setStyleSheet("""
        QPushButton {
            background-color: #48bb78;
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 6px;
            font-weight: 600;
        }
        QPushButton:hover {
            background-color: #38a169;
        }
        QPushButton:pressed {
            background-color: #2f855a;
        }
    """)
    app.today_btn.clicked.connect(app.go_to_today)
    date_nav_layout.addWidget(app.today_btn)

    layout.addWidget(date_nav_widget)

    # === SUBTLE ALERT BANNER ===
    app.alert_banner = QLabel()
    app.alert_banner.setWordWrap(True)
    app.alert_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
    app.alert_banner.setFont(QFont("SF Pro Text", 13, QFont.Weight.Bold))
    app.alert_banner.setStyleSheet("""
        background-color: #fff5f5;
        color: #e74c3c;
        padding: 10px;
        border-radius: 6px;
        border-left: 4px solid #e74c3c;
    """)
    app.alert_banner.hide()  # Hidden by default
    layout.addWidget(app.alert_banner)

    # === CURRENT MOOD DISPLAY ===
    mood_widget = QWidget()
    mood_widget.setStyleSheet("""
        QWidget {
            background-color: white;
            border-radius: 12px;
            border: 1px solid #e1e8ed;
        }
    """)
    mood_layout = QVBoxLayout()
    mood_layout.setSpacing(10)
    mood_layout.setContentsMargins(30, 20, 30, 20)
    mood_widget.setLayout(mood_layout)

    app.mood_emoji = QLabel("😐")
    app.mood_emoji.setFont(QFont("Apple Color Emoji", 64))
    app.mood_emoji.setAlignment(Qt.AlignmentFlag.AlignCenter)
    mood_layout.addWidget(app.mood_emoji)

    app.mood_label = QLabel("Current Mood: Neutral (0.00)")
    app.mood_label.setFont(QFont("SF Pro Text", 18, QFont.Weight.DemiBold))
    app.mood_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    app.mood_label.setStyleSheet("color: #34495e; padding: 5px;")
    mood_layout.addWidget(app.mood_label)

    app.mood_bar = QProgressBar()
    app.mood_bar.setMinimum(-100)
    app.mood_bar.setMaximum(100)
    app.mood_bar.setValue(0)
    app.mood_bar.setTextVisible(True)
    app.mood_bar.setFormat("%v% Mood Score")
    app.mood_bar.setFixedHeight(35)
    mood_layout.addWidget(app.mood_bar)

    layout.addWidget(mood_widget)

    # === TABS ===
    tabs = QTabWidget()

    # Tab 1: Dashboard (NEW!)
    dashboard_tab = QWidget()
    dashboard_layout = QVBoxLayout()
    dashboard_layout.setSpacing(15)
    dashboard_tab.setLayout(dashboard_layout)

    # Dashboard cards container
    cards_container = QWidget()
    cards_layout = QHBoxLayout()
    cards_layout.setSpacing(15)
    cards_container.setLayout(cards_layout)

    # Card 1: Work Hours
    work_card = QWidget()
    work_card.setStyleSheet("""
        QWidget {
            background-color: white;
            border-radius: 10px;
            border: 1px solid #e1e8ed;
            padding: 15px;
        }
    """)
    work_card_layout = QVBoxLayout()
    work_card.setLayout(work_card_layout)

    work_icon = QLabel("⏰")
    work_icon.setFont(QFont("Apple Color Emoji", 32))
    work_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
    work_card_layout.addWidget(work_icon)

    app.dashboard_work_hours = QLabel("0.0h")
    app.dashboard_work_hours.setFont(QFont("SF Pro Display", 28, QFont.Weight.Bold))
    app.dashboard_work_hours.setAlignment(Qt.AlignmentFlag.AlignCenter)
    app.dashboard_work_hours.setStyleSheet("color: #3498db;")
    work_card_layout.addWidget(app.dashboard_work_hours)

    work_label = QLabel("Work Hours")
    work_label.setFont(QFont("SF Pro Text", 12))
    work_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    work_label.setStyleSheet("color: #7f8c8d;")
    work_card_layout.addWidget(work_label)

    cards_layout.addWidget(work_card)

    # Card 2: Breaks
    break_card = QWidget()
    break_card.setStyleSheet("""
        QWidget {
            background-color: white;
            border-radius: 10px;
            border: 1px solid #e1e8ed;
            padding: 15px;
        }
    """)
    break_card_layout = QVBoxLayout()
    break_card.setLayout(break_card_layout)

    break_icon = QLabel("☕")
    break_icon.setFont(QFont("Apple Color Emoji", 32))
    break_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
    break_card_layout.addWidget(break_icon)

    app.dashboard_breaks = QLabel("0")
    app.dashboard_breaks.setFont(QFont("SF Pro Display", 28, QFont.Weight.Bold))
    app.dashboard_breaks.setAlignment(Qt.AlignmentFlag.AlignCenter)
    app.dashboard_breaks.setStyleSheet("color: #e67e22;")
    break_card_layout.addWidget(app.dashboard_breaks)

    break_label = QLabel("Breaks Taken")
    break_label.setFont(QFont("SF Pro Text", 12))
    break_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    break_label.setStyleSheet("color: #7f8c8d;")
    break_card_layout.addWidget(break_label)

    cards_layout.addWidget(break_card)

    # Card 3: Messages
    msg_card = QWidget()
    msg_card.setStyleSheet("""
        QWidget {
            background-color: white;
            border-radius: 10px;
            border: 1px solid #e1e8ed;
            padding: 15px;
        }
    """)
    msg_card_layout = QVBoxLayout()
    msg_card.setLayout(msg_card_layout)

    msg_icon = QLabel("💬")
    msg_icon.setFont(QFont("Apple Color Emoji", 32))
    msg_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
    msg_card_layout.addWidget(msg_icon)

    app.dashboard_messages = QLabel("0")
    app.dashboard_messages.setFont(QFont("SF Pro Display", 28, QFont.Weight.Bold))
    app.dashboard_messages.setAlignment(Qt.AlignmentFlag.AlignCenter)
    app.dashboard_messages.setStyleSheet("color: #9b59b6;")
    msg_card_layout.addWidget(app.dashboard_messages)

    msg_label = QLabel("Messages Today")
    msg_label.setFont(QFont("SF Pro Text", 12))
    msg_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    msg_label.setStyleSheet("color: #7f8c8d;")
    msg_card_layout.addWidget(msg_label)

    cards_layout.addWidget(msg_card)

    # Card 4: Health Status
    health_card = QWidget()
    health_card.setStyleSheet("""
        QWidget {
            background-color: white;
            border-radius: 10px;
            border: 1px solid #e1e8ed;
            padding: 15px;
        }
    """)
    health_card_layout = QVBoxLayout()
    health_card.setLayout(health_card_layout)

    app.dashboard_health_emoji = QLabel("✅")
    app.dashboard_health_emoji.setFont(QFont("Apple Color Emoji", 32))
    app.dashboard_health_emoji.setAlignment(Qt.AlignmentFlag.AlignCenter)
    health_card_layout.addWidget(app.dashboard_health_emoji)

    app.dashboard_health_status = QLabel("Healthy")
    app.dashboard_health_status.setFont(QFont("SF Pro Display", 20, QFont.Weight.Bold))
    app.dashboard_health_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
    app.dashboard_health_status.setStyleSheet("color: #27ae60;")
    health_card_layout.addWidget(app.dashboard_health_status)

    health_label = QLabel("Work Health")
    health_label.setFont(QFont("SF Pro Text", 12))
    health_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    health_label.setStyleSheet("color: #7f8c8d;")
    health_card_layout.addWidget(health_label)

    cards_layout.addWidget(health_card)

    dashboard_layout.addWidget(cards_container)

    # Dashboard summary section
    app.dashboard_summary = QLabel("Loading dashboard...")
    app.dashboard_summary.setWordWrap(True)
    app.dashboard_summary.setFont(QFont("SF Pro Text", 13))
    app.dashboard_summary.setStyleSheet("""
        QLabel {
            background-color: white;
            border-radius: 10px;
            border: 1px solid #e1e8ed;
            padding: 20px;
        }
    """)
    dashboard_layout.addWidget(app.dashboard_summary)

    tabs.addTab(dashboard_tab, "📊 Dashboard")

    # Tab 2: Live Feed
    live_tab = QWidget()
    live_layout = QVBoxLayout()
    live_tab.setLayout(live_layout)

    live_layout.addWidget(QLabel("Recent Messages:"))
    app.live_feed = QTextEdit()
    app.live_feed.setReadOnly(True)
    app.live_feed.setMaximumHeight(200)
    live_layout.addWidget(app.live_feed)

    tabs.addTab(live_tab, "💬 Live Feed")

    # Tab 3: Chart & Insights
    chart_tab = QWidget()
    chart_layout = QVBoxLayout()
    chart_layout.setContentsMargins(0, 0, 0, 0)
    chart_tab.setLayout(chart_layout)

    # Time tracking info
    app.time_info_label = QLabel("Loading work session info...")
    app.time_info_label.setWordWrap(True)
    app.time_info_label.setFont(QFont("Arial", 11))
    app.time_info_label.setStyleSheet("padding: 10px; background-color: #f8f9fa; border-radius: 6px; margin: 5px;")
    chart_layout.addWidget(app.time_info_label)

    # Chart in scroll area
    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll_area.setStyleSheet("""
        QScrollArea {
            border: none;
            background-color: #f8f9fa;
        }
    """)

    app.chart_widget = SentimentChartWidget()
    app.chart_widget.setMinimumHeight(800)  # Give charts room to breathe
    scroll_area.setWidget(app.chart_widget)
    chart_layout.addWidget(scroll_area)

    tabs.addTab(chart_tab, "📈 Chart & Insights")

    # Tab 4: Health (Phase 1)
    health_tab = QWidget()
    health_layout = QVBoxLayout()
    health_tab.setLayout(health_layout)

    app.health_label = QLabel("Loading health metrics...")
    app.health_label.setWordWrap(True)
    app.health_label.setFont(QFont("Arial", 11))
    health_layout.addWidget(app.health_label)

    tabs.addTab(health_tab, "🏥 Health")

    # Tab 5: Statistics
    stats_tab = QWidget()
    stats_layout = QVBoxLayout()
    stats_tab.setLayout(stats_layout)

    app.stats_label = QLabel("Loading statistics...")
    app.stats_label.setWordWrap(True)
    stats_layout.addWidget(app.stats_label)

    tabs.addTab(stats_tab, "📊 Statistics")

    # Tab 6: Alerts
    alerts_tab = QWidget()
    alerts_layout = QVBoxLayout()
    alerts_tab.setLayout(alerts_layout)

    app.alerts_label = QLabel("No alerts yet")
    app.alerts_label.setWordWrap(True)
    alerts_layout.addWidget(app.alerts_label)

    tabs.addTab(alerts_tab, "⚠️ Alerts")

    # Tab 5: Background Tasks
    tasks_tab = QWidget()
    tasks_layout = QVBoxLayout()
    tasks_layout.setContentsMargins(20, 20, 20, 20)
    tasks_tab.setLayout(tasks_layout)

    # Title
    tasks_title = QLabel("🔄 Background Tasks")
    tasks_title.setFont(QFont("SF Pro Display", 18, QFont.Weight.Bold))
    tasks_title.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
    tasks_layout.addWidget(tasks_title)

    # Description
    tasks_desc = QLabel("Analysis tasks running in the background")
    tasks_desc.setStyleSheet("color: #7f8c8d; margin-bottom: 20px;")
    tasks_layout.addWidget(tasks_desc)

    # Analysis Progress Widget
    app.analysis_progress_widget = QWidget()
    app.analysis_progress_widget.setStyleSheet("""
        QWidget {
            background-color: #e3f2fd;
            border-radius: 8px;
            padding: 20px;
        }
    """)
    analysis_progress_layout = QVBoxLayout()
    analysis_progress_layout.setContentsMargins(15, 15, 15, 15)
    app.analysis_progress_widget.setLayout(analysis_progress_layout)

    app.analysis_progress_label = QLabel("💤 No tasks running")
    app.analysis_progress_label.setFont(QFont("SF Pro Text", 13))
    app.analysis_progress_label.setStyleSheet("color: #1976d2; background: transparent;")
    analysis_progress_layout.addWidget(app.analysis_progress_label)

    app.analysis_progress_bar = QProgressBar()
    app.analysis_progress_bar.setStyleSheet("""
        QProgressBar {
            border: 2px solid #90caf9;
            border-radius: 5px;
            text-align: center;
            background-color: white;
            height: 30px;
            font-size: 13px;
        }
        QProgressBar::chunk {
            background-color: #42a5f5;
            border-radius: 3px;
        }
    """)
    app.analysis_progress_bar.setMinimum(0)
    app.analysis_progress_bar.setMaximum(100)
    app.analysis_progress_bar.setValue(0)
    app.analysis_progress_bar.setFormat("%v / %m messages")
    app.analysis_progress_bar.hide()  # Hidden until task starts
    analysis_progress_layout.addWidget(app.analysis_progress_bar)

    tasks_layout.addWidget(app.analysis_progress_widget)
    tasks_layout.addStretch()

    tabs.addTab(tasks_tab, "🔄 Tasks")

    layout.addWidget(tabs)

    # === FOOTER BUTTONS ===
    button_layout = QHBoxLayout()
    button_layout.setSpacing(10)

    refresh_btn = QPushButton("🔄 Refresh")
    refresh_btn.setStyleSheet("""
        QPushButton {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
        }
        QPushButton:hover {
            background-color: #2980b9;
        }
    """)
    refresh_btn.clicked.connect(app.load_recent_conversations)
    button_layout.addWidget(refresh_btn)

    export_btn = QPushButton("💾 Export Data")
    export_btn.setStyleSheet("""
        QPushButton {
            background-color: #27ae60;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
        }
        QPushButton:hover {
            background-color: #229954;
        }
    """)
    export_btn.clicked.connect(app.export_data)
    button_layout.addWidget(export_btn)

    quit_btn = QPushButton("❌ Quit")
    quit_btn.setStyleSheet("""
        QPushButton {
            background-color: #e74c3c;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
        }
        QPushButton:hover {
            background-color: #c0392b;
        }
    """)
    quit_btn.clicked.connect(app.close)
    button_layout.addWidget(quit_btn)

    layout.addLayout(button_layout)
