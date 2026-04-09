from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTabBar,
    QTextEdit,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
)


class LibraryPanel(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Library", parent)

        layout = QVBoxLayout(self)

        self.prompt_library_hint = QLabel(
            "Click to load. Double-click to send or run. Right-click to rename or edit tags."
        )
        self.prompt_library_hint.setWordWrap(True)
        self.prompt_library_hint.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.prompt_library_hint.setStyleSheet("QLabel { color: #cbd5e1; padding: 0 0 4px 0; }")

        self.library_nav_row = QWidget()
        self.library_nav_layout = QHBoxLayout(self.library_nav_row)
        self.library_nav_layout.setContentsMargins(0, 0, 0, 0)
        self.library_nav_layout.setSpacing(6)

        self.library_tabs = QTabBar()
        self.library_tabs.setDrawBase(False)
        self.library_tabs.addTab("Prompts")
        self.library_tabs.addTab("Code")
        self.library_tabs.addTab("Templates")
        self.library_tabs.setExpanding(False)
        self.library_tabs.setMovable(False)
        self.library_tabs.setDocumentMode(True)
        self.library_tabs.setElideMode(Qt.ElideRight)
        self.library_tabs.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.library_tabs.setTabToolTip(0, "Reusable natural-language prompts you can load, save, pin, and send.")
        self.library_tabs.setTabToolTip(1, "Saved or recent Python snippets for Run My Code and code refinement.")
        self.library_tabs.setTabToolTip(2, "Built-in starter templates organized by category for loading or immediate execution.")

        self.actions_tab_btn = QPushButton("Actions")
        self.actions_tab_btn.setCheckable(True)
        self.actions_tab_btn.setToolTip("Deterministic built-in actions you can preview, load into Prompt, or run directly.")

        self.library_nav_layout.addWidget(self.library_tabs, 0)
        self.library_nav_layout.addStretch(1)
        self.library_nav_layout.addWidget(self.actions_tab_btn, 0)

        self.library_stack = QStackedWidget()
        self.library_stack.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)

        self.prompt_library_list = QListWidget()
        self.prompt_library_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.prompt_library_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.prompt_library_list.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.prompt_library_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.prompt_library_list.setWordWrap(True)
        self.prompt_library_list.setStyleSheet(
            "QListWidget { background: #101820; color: #d6deeb; border: 1px solid #22304a; } "
            "QListWidget::item:selected { background: #1d2a44; color: #e8f1ff; }"
        )

        self.code_library_list = QListWidget()
        self.code_library_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.code_library_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.code_library_list.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.code_library_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.code_library_list.setWordWrap(True)
        self.code_library_list.setStyleSheet(
            "QListWidget { background: #101820; color: #d6deeb; border: 1px solid #22304a; } "
            "QListWidget::item:selected { background: #1d2a44; color: #e8f1ff; }"
        )

        prompt_library_font = self.prompt_library_list.font()
        if prompt_library_font.pointSize() > 0:
            prompt_library_font.setPointSize(prompt_library_font.pointSize() + 1)
            self.prompt_library_list.setFont(prompt_library_font)
            self.code_library_list.setFont(prompt_library_font)

        self.template_tab = QWidget()
        template_tab_layout = QVBoxLayout(self.template_tab)
        template_tab_layout.setContentsMargins(0, 0, 0, 0)
        self.template_splitter = QSplitter(Qt.Horizontal)
        self.template_tree = QTreeWidget()
        self.template_tree.setHeaderHidden(True)
        self.template_tree.setMinimumWidth(180)
        self.template_tree.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.template_tree.setStyleSheet("QTreeWidget { background: #101820; color: #d6deeb; border: 1px solid #22304a; }")
        self.template_preview = QTextEdit()
        self.template_preview.setReadOnly(True)
        self.template_preview.setAcceptRichText(False)
        self.template_preview.setPlaceholderText("Select a template to preview its description and code.")
        self.template_preview.setStyleSheet(
            "QTextEdit { background: #0b1021; color: #d6deeb; border: 1px solid #22304a; padding: 10px; }"
        )
        self.template_splitter.addWidget(self.template_tree)
        self.template_splitter.addWidget(self.template_preview)
        self.template_splitter.setStretchFactor(0, 0)
        self.template_splitter.setStretchFactor(1, 1)
        template_tab_layout.addWidget(self.template_splitter, 1)
        self.template_btn_row = QWidget()
        template_btn_layout = QHBoxLayout(self.template_btn_row)
        self.template_load_btn = QPushButton("Load Template")
        self.template_load_btn.setToolTip("Load the selected built-in template into the Prompt box for refinement.")
        self.template_run_btn = QPushButton("Run Template")
        self.template_run_btn.setToolTip("Load the selected built-in template and run it immediately with Run My Code.")
        template_btn_layout.addWidget(self.template_load_btn)
        template_btn_layout.addWidget(self.template_run_btn)
        template_tab_layout.addWidget(self.template_btn_row)

        self.action_tab = QWidget()
        action_tab_layout = QVBoxLayout(self.action_tab)
        action_tab_layout.setContentsMargins(0, 0, 0, 0)
        self.action_splitter = QSplitter(Qt.Horizontal)
        self.action_tree = QTreeWidget()
        self.action_tree.setHeaderHidden(True)
        self.action_tree.setMinimumWidth(180)
        self.action_tree.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.action_tree.setStyleSheet("QTreeWidget { background: #101820; color: #d6deeb; border: 1px solid #22304a; }")
        self.action_preview = QTextEdit()
        self.action_preview.setReadOnly(True)
        self.action_preview.setAcceptRichText(False)
        self.action_preview.setPlaceholderText("Select an action to preview what it does and how it runs.")
        self.action_preview.setStyleSheet(
            "QTextEdit { background: #0b1021; color: #d6deeb; border: 1px solid #22304a; padding: 10px; }"
        )
        self.action_splitter.addWidget(self.action_tree)
        self.action_splitter.addWidget(self.action_preview)
        self.action_splitter.setStretchFactor(0, 0)
        self.action_splitter.setStretchFactor(1, 1)
        action_tab_layout.addWidget(self.action_splitter, 1)
        self.action_btn_row = QWidget()
        action_btn_layout = QHBoxLayout(self.action_btn_row)
        self.action_load_btn = QPushButton("Load Action")
        self.action_load_btn.setToolTip("Load the selected action's suggested prompt into the Prompt box.")
        self.action_run_btn = QPushButton("Run Action")
        self.action_run_btn.setToolTip("Run the selected deterministic action directly without using the model.")
        self.action_add_shortcut_btn = QPushButton("Add to Shortcuts")
        self.action_add_shortcut_btn.setToolTip("Add the selected action to the Shortcuts area for one-click reuse.")
        action_btn_layout.addWidget(self.action_load_btn)
        action_btn_layout.addWidget(self.action_run_btn)
        action_btn_layout.addWidget(self.action_add_shortcut_btn)
        action_tab_layout.addWidget(self.action_btn_row)

        self.library_stack.addWidget(self.prompt_library_list)
        self.library_stack.addWidget(self.code_library_list)
        self.library_stack.addWidget(self.template_tab)
        self.library_stack.addWidget(self.action_tab)

        self.prompt_library_btn_row = QWidget()
        prompt_library_btn_layout = QHBoxLayout(self.prompt_library_btn_row)
        self.save_prompt_btn = QPushButton("Save")
        self.pin_prompt_btn = QPushButton("Pin")
        self.delete_prompt_btn = QPushButton("Delete")
        self.clear_prompt_btn = QPushButton("Clear")
        self.prompt_font_down_btn = QPushButton("A-")
        self.prompt_font_down_btn.setToolTip("Decrease library font size.")
        self.prompt_font_up_btn = QPushButton("A+")
        self.prompt_font_up_btn.setToolTip("Increase library font size.")
        prompt_library_btn_layout.addWidget(self.save_prompt_btn)
        prompt_library_btn_layout.addWidget(self.pin_prompt_btn)
        prompt_library_btn_layout.addWidget(self.delete_prompt_btn)
        prompt_library_btn_layout.addWidget(self.clear_prompt_btn)
        prompt_library_btn_layout.addWidget(self.prompt_font_down_btn)
        prompt_library_btn_layout.addWidget(self.prompt_font_up_btn)

        layout.addWidget(self.prompt_library_hint)
        layout.addWidget(self.library_nav_row)
        layout.addWidget(self.library_stack)
        layout.addWidget(self.prompt_library_btn_row)
