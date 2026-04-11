from __future__ import annotations

from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QColor, QIcon, QPainter, QPixmap
from qtpy.QtWidgets import QHBoxLayout, QLabel, QMenu, QPushButton, QVBoxLayout, QWidget


def _make_stop_icon() -> QIcon:
    pixmap = QPixmap(12, 12)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, False)
    painter.fillRect(1, 1, 10, 10, QColor("#d32f2f"))
    painter.end()
    return QIcon(pixmap)


class PendingCodePanel(QWidget):
    def __init__(self, *, ui_help_enabled: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        top_row = QWidget()
        top_row_layout = QHBoxLayout(top_row)
        top_row_layout.setContentsMargins(0, 0, 0, 0)
        top_row_layout.setSpacing(6)

        self.pending_code_label = QLabel("Pending code: none")
        self.pending_code_label.setStyleSheet("QLabel { color: #c7d2fe; padding: 2px 0; }")

        self.help_btn = QPushButton("Help")
        self.help_btn.setToolTip("Open help, version notes, bug reporting, and UI-help controls.")
        self.help_menu = QMenu(self.help_btn)
        self.help_whats_new_action = self.help_menu.addAction("What's New")
        self.help_about_action = self.help_menu.addAction("About")
        self.help_report_bug_action = self.help_menu.addAction("Report Bug")
        self.help_menu.addSeparator()
        self.help_ui_toggle_action = self.help_menu.addAction("UI Help Enabled")
        self.help_ui_toggle_action.setCheckable(True)
        self.help_ui_toggle_action.setChecked(bool(ui_help_enabled))
        self.help_btn.setMenu(self.help_menu)

        self.advanced_btn = QPushButton("Advanced")
        self.advanced_btn.setToolTip("Open advanced and optional integrations.")
        self.advanced_menu = QMenu(self.advanced_btn)
        self.sam2_setup_action = self.advanced_menu.addAction("SAM2 Setup")
        self.sam2_live_action = self.advanced_menu.addAction("SAM2 Live")
        self.text_annotation_action = self.advanced_menu.addAction("Text Annotation")
        self.atlas_stitch_action = self.advanced_menu.addAction("Atlas Stitch")
        self.advanced_btn.setMenu(self.advanced_menu)

        self.run_code_btn = QPushButton("Run Code")
        self.run_my_code_btn = QPushButton("Run My Code")
        self.run_code_btn.setToolTip(
            "Run the reviewed code inside the Chat Assistant plugin runtime. "
            "This uses plugin globals such as viewer, selected_layer, and run_in_background, "
            "and output is shown in chat rather than QtConsole."
        )
        self.run_my_code_btn.setToolTip(
            "Paste your own Python in the Prompt box and run it inside the Chat Assistant plugin runtime, "
            "without opening QtConsole. This is similar to napari scripting but not identical to QtConsole. "
            "Output is shown in chat."
        )

        self.refine_my_code_btn = QPushButton("Refine My Code")
        self.refine_my_code_btn.setToolTip(
            "Ask the assistant to adapt prompt-box code or the last failed Run My Code submission so it works "
            "in the Chat Assistant plugin runtime. Use this when code may work in QtConsole but not here as-is."
        )

        self.copy_code_btn = QPushButton("Copy Code")
        self.run_code_btn.setEnabled(False)
        self.copy_code_btn.setEnabled(False)
        self.refine_my_code_btn.setEnabled(False)

        self.chat_font_down_btn = QPushButton("A-")
        self.chat_font_down_btn.setToolTip("Decrease chat font size.")
        self.chat_font_up_btn = QPushButton("A+")
        self.chat_font_up_btn.setToolTip("Increase chat font size.")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setIcon(_make_stop_icon())
        self.stop_btn.setIconSize(QSize(12, 12))
        self.stop_btn.setToolTip("Stop the current model request.")
        self.feedback_btn = QPushButton("Rate Result")
        self.feedback_btn.setToolTip("Rate the latest assistant result for local quality and routing improvement.")
        self.feedback_menu = QMenu(self.feedback_btn)
        self.feedback_helpful_action = self.feedback_menu.addAction("Helpful")
        self.feedback_wrong_route_action = self.feedback_menu.addAction("Wrong Route")
        self.feedback_wrong_answer_action = self.feedback_menu.addAction("Wrong Answer")
        self.feedback_didnt_work_action = self.feedback_menu.addAction("Didn't Work")
        self.feedback_btn.setMenu(self.feedback_menu)

        bottom_row = QWidget()
        bottom_row_layout = QHBoxLayout(bottom_row)
        bottom_row_layout.setContentsMargins(0, 0, 0, 0)
        bottom_row_layout.setSpacing(6)

        top_row_layout.addWidget(self.pending_code_label, 1)
        top_row_layout.addWidget(self.run_code_btn)
        top_row_layout.addWidget(self.copy_code_btn)
        top_row_layout.addWidget(self.run_my_code_btn)
        top_row_layout.addWidget(self.refine_my_code_btn)

        bottom_row_layout.addStretch(1)
        bottom_row_layout.addWidget(self.stop_btn)
        bottom_row_layout.addWidget(self.chat_font_down_btn)
        bottom_row_layout.addWidget(self.chat_font_up_btn)
        bottom_row_layout.addWidget(self.feedback_btn)
        bottom_row_layout.addWidget(self.advanced_btn)
        bottom_row_layout.addWidget(self.help_btn)

        layout.addWidget(top_row)
        layout.addWidget(bottom_row)
