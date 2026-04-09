from __future__ import annotations

from qtpy.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ShortcutsPanel(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Shortcuts", parent)

        self.shortcuts_layout = QVBoxLayout(self)

        self.shortcuts_hint = QLabel(
            "Keep your most-used actions here for one-click work."
        )
        self.shortcuts_hint.setWordWrap(True)
        self.shortcuts_hint.setStyleSheet(
            "QLabel { color: #cbd5e1; padding: 0 0 4px 0; }"
        )
        self.shortcuts_layout.addWidget(self.shortcuts_hint)

        self.shortcuts_grid = QGridLayout()
        self.shortcuts_grid.setContentsMargins(0, 0, 0, 0)
        self.shortcuts_grid.setHorizontalSpacing(8)
        self.shortcuts_grid.setVerticalSpacing(8)
        self.shortcuts_layout.addLayout(self.shortcuts_grid)

        self.shortcuts_btn_row = QWidget()
        shortcuts_btn_layout = QHBoxLayout(self.shortcuts_btn_row)
        self.shortcuts_add_row_btn = QPushButton("+")
        self.shortcuts_add_row_btn.setToolTip(
            "Add another row of 3 shortcut buttons."
        )
        self.shortcuts_add_row_btn.setFixedWidth(88)
        self.shortcuts_remove_row_btn = QPushButton("-")
        self.shortcuts_remove_row_btn.setToolTip(
            "Remove the last row of 3 shortcut buttons. The last row must be empty first."
        )
        self.shortcuts_remove_row_btn.setFixedWidth(88)
        self.shortcuts_save_btn = QPushButton("Save Setup")
        self.shortcuts_save_btn.setToolTip(
            "Save your current shortcuts setup so you can reuse it later."
        )
        self.shortcuts_save_btn.setFixedWidth(88)
        self.shortcuts_load_btn = QPushButton("Load Setup")
        self.shortcuts_load_btn.setToolTip("Load a saved shortcuts setup.")
        self.shortcuts_load_btn.setFixedWidth(88)
        self.shortcuts_clear_btn = QPushButton("Clear")
        self.shortcuts_clear_btn.setToolTip(
            "Clear all shortcut button assignments."
        )
        self.shortcuts_clear_btn.setFixedWidth(88)
        shortcuts_btn_layout.addStretch(1)
        shortcuts_btn_layout.addWidget(self.shortcuts_add_row_btn)
        shortcuts_btn_layout.addWidget(self.shortcuts_remove_row_btn)
        shortcuts_btn_layout.addWidget(self.shortcuts_save_btn)
        shortcuts_btn_layout.addWidget(self.shortcuts_load_btn)
        shortcuts_btn_layout.addWidget(self.shortcuts_clear_btn)
        self.shortcuts_layout.addWidget(self.shortcuts_btn_row)
