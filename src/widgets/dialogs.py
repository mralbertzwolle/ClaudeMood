"""
Dialogs for ClaudeMood
"""
from pathlib import Path
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QListWidget, QDialogButtonBox
)


class DirectorySelectionDialog(QDialog):
    """Dialog for selecting conversations directory"""

    def __init__(self, directories, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Conversations Directory")
        self.setMinimumWidth(600)
        self.setMinimumHeight(300)

        self.selected_directory = None

        layout = QVBoxLayout()

        # Title
        title = QLabel("🔍 Multiple conversations directories found!\n\nSelect the correct directory:")
        title.setWordWrap(True)
        layout.addWidget(title)

        # Directory list
        self.list_widget = QListWidget()
        for directory in directories:
            dir_path = Path(directory)
            # Count conversation files (JSON or JSONL)
            if dir_path.name == "projects":
                file_count = sum(1 for _ in dir_path.rglob("*.jsonl"))
            else:
                file_count = len(list(dir_path.glob("*.json")))
            self.list_widget.addItem(f"{directory} ({file_count} conversations)")

        if directories:
            self.list_widget.setCurrentRow(0)

        layout.addWidget(self.list_widget)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)
        self.directories = directories

    def accept(self):
        """User clicked OK"""
        current_row = self.list_widget.currentRow()
        if current_row >= 0:
            self.selected_directory = self.directories[current_row]
        super().accept()
