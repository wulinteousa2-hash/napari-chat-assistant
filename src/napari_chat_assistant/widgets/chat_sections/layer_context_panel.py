from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QGroupBox,
    QListWidget,
    QSizePolicy,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class LayerContextPanel(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Layer Context", parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        self.context_tabs = QTabWidget()
        layout.addWidget(self.context_tabs)

        context_summary_tab = QWidget()
        context_summary_layout = QVBoxLayout(context_summary_tab)
        context_summary_layout.setContentsMargins(0, 0, 0, 0)
        self.context_summary_box = QTextEdit()
        self.context_summary_box.setReadOnly(True)
        self.context_summary_box.setAcceptRichText(False)
        self.context_summary_box.setPlaceholderText("Copyable layer context will appear here.")
        self.context_summary_box.setMinimumHeight(120)
        self.context_summary_box.setMaximumHeight(180)
        self.context_summary_box.setStyleSheet(
            "QTextEdit { background: #101820; color: #e6edf3; border: 1px solid #22304a; padding: 8px; }"
        )
        context_summary_layout.addWidget(self.context_summary_box)

        context_layers_tab = QWidget()
        context_layers_layout = QVBoxLayout(context_layers_tab)
        context_layers_layout.setContentsMargins(0, 0, 0, 0)
        self.context_selected_only_checkbox = QCheckBox("Show selected layer(s) only")
        self.context_selected_only_checkbox.setToolTip("Hide all non-selected layers. Turn off to restore the previous visibility state.")
        context_layers_layout.addWidget(self.context_selected_only_checkbox)
        self.context_layers_list = QListWidget()
        self.context_layers_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.context_layers_list.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.context_layers_list.setStyleSheet(
            "QListWidget { background: #101820; color: #d6deeb; border: 1px solid #22304a; }"
            "QListWidget::item:selected { background: #1d3b5f; color: #ffffff; }"
        )
        context_layers_layout.addWidget(self.context_layers_list)

        self.context_tabs.addTab(context_summary_tab, "Summary")
        self.context_tabs.addTab(context_layers_tab, "Layers")
