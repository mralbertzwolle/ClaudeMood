"""
Directory Finder - Handles finding and selecting conversations directory
"""
from pathlib import Path
from PyQt6.QtWidgets import QMessageBox, QDialog, QProgressDialog, QApplication
from PyQt6.QtCore import Qt
from widgets import DirectorySelectionDialog
from workers import SearchWorker
from config_utils import save_config


class DirectoryFinder:
    """Finds and selects conversations directory"""

    def __init__(self, parent, config):
        self.parent = parent
        self.config = config

    def find_and_select_directory(self, conversations_dir):
        """
        Find conversations directory or let user select one

        Returns:
            Path or None: Selected directory or None if cancelled
        """
        if conversations_dir.exists():
            return conversations_dir

        print(f"⚠️ Directory not found: {conversations_dir}")

        # Search for directories
        found_dirs = self._search_for_directories()

        if not found_dirs:
            QMessageBox.warning(
                self.parent,
                "Directory Not Found",
                f"Conversations directory not found:\n{conversations_dir}\n\n"
                "Could not find any conversations directories automatically.\n\n"
                "Please check your Claude Code installation."
            )
            return None

        # Auto-select if only one found
        if len(found_dirs) == 1:
            selected = found_dirs[0]
            print(f"✅ Auto-selected: {selected}")
            self._update_config(selected)
            return selected

        # Multiple directories - let user choose
        return self._show_selection_dialog(found_dirs)

    def _search_for_directories(self):
        """Search for conversations directories with progress dialog"""
        progress = QProgressDialog("Searching for conversations directories...", "Cancel", 0, 0, self.parent)
        progress.setWindowTitle("ClaudeMood - Searching...")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setCancelButton(None)
        progress.setAutoClose(False)
        progress.show()
        QApplication.processEvents()

        # Start search worker
        search_worker = SearchWorker()
        found_dirs = []

        def on_progress(message):
            progress.setLabelText(message)
            QApplication.processEvents()

        def on_finished(dirs):
            nonlocal found_dirs
            found_dirs = dirs
            progress.close()

        search_worker.progress.connect(on_progress)
        search_worker.finished.connect(on_finished)
        search_worker.start()

        # Wait for search
        while search_worker.isRunning():
            QApplication.processEvents()

        return found_dirs

    def _show_selection_dialog(self, found_dirs):
        """Show dialog for user to select directory"""
        dialog = DirectorySelectionDialog(found_dirs, self.parent)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.selected_directory:
            selected = Path(dialog.selected_directory)
            print(f"✅ User selected: {selected}")
            self._update_config(selected)
            return selected
        else:
            QMessageBox.warning(
                self.parent,
                "No Directory Selected",
                "No conversations directory selected. Monitoring disabled."
            )
            return None

    def _update_config(self, directory):
        """Update config with selected directory"""
        self.config['conversations_dir'] = directory
        save_config(self.config)
        print("💾 Config updated")
