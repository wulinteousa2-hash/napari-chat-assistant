from __future__ import annotations

import napari
import numpy as np
from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from napari_chat_assistant.agent.context import find_labels_layer
from napari_chat_assistant.agent.dispatcher import apply_tool_job_result, prepare_tool_job, run_tool_job


WIDGET_NAME = "Relabel Mask Values"


def _dock_wrapper_from_widget(widget):
    try:
        return widget.parent()
    except Exception:
        pass
    try:
        return widget.native.parent()
    except Exception:
        return None


class RelabelMaskWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.setMinimumWidth(360)
        self.setWindowTitle(WIDGET_NAME)

        layout = QVBoxLayout(self)

        self.status_label = QLabel("Select a Labels layer, choose the source value, and set the new value.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        form = QFormLayout()

        self.layer_combo = QComboBox()
        self.source_combo = QComboBox()
        self.source_combo.setEditable(True)
        self.target_spin = QSpinBox()
        self.target_spin.setRange(0, 2_147_483_647)
        self.target_spin.setValue(5)

        form.addRow("Labels Layer", self.layer_combo)
        form.addRow("Source Value", self.source_combo)
        form.addRow("Target Value", self.target_spin)
        layout.addLayout(form)

        self.values_label = QLabel("Available values: []")
        self.values_label.setWordWrap(True)
        layout.addWidget(self.values_label)

        controls = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh")
        self.run_btn = QPushButton("Relabel")
        controls.addWidget(self.refresh_btn)
        controls.addWidget(self.run_btn)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.layer_combo.currentTextChanged.connect(self._refresh_value_choices)
        self.refresh_btn.clicked.connect(self.refresh_from_viewer)
        self.run_btn.clicked.connect(self.run_relabel)

        self.refresh_from_viewer()

    def _labels_layer_names(self) -> list[str]:
        names: list[str] = []
        for layer in list(self.viewer.layers):
            try:
                if layer.__class__.__name__ == "Labels":
                    names.append(str(layer.name))
            except Exception:
                continue
        return names

    def _current_layer(self):
        name = str(self.layer_combo.currentText() or "").strip()
        return find_labels_layer(self.viewer, name)

    def refresh_from_viewer(self):
        names = self._labels_layer_names()
        current_selected = getattr(self.viewer.layers.selection, "active", None)
        selected_name = ""
        if current_selected is not None and current_selected.__class__.__name__ == "Labels":
            selected_name = str(getattr(current_selected, "name", "") or "")

        previous = str(self.layer_combo.currentText() or "").strip()
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        self.layer_combo.addItems(names)
        preferred = selected_name or previous
        if preferred and preferred in names:
            self.layer_combo.setCurrentText(preferred)
        self.layer_combo.blockSignals(False)
        self._refresh_value_choices()

    def _refresh_value_choices(self):
        layer = self._current_layer()
        if layer is None:
            self.source_combo.clear()
            self.values_label.setText("Available values: []")
            self.status_label.setText("No Labels layer is selected. Select or create a labels layer first.")
            return
        data = np.asarray(layer.data)
        unique_values = [int(v) for v in np.unique(data).tolist()]
        current_source = str(self.source_combo.currentText() or "").strip()
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        self.source_combo.addItems([str(v) for v in unique_values])
        if current_source:
            self.source_combo.setCurrentText(current_source)
        elif 1 in unique_values:
            self.source_combo.setCurrentText("1")
        elif unique_values:
            self.source_combo.setCurrentText(str(unique_values[0]))
        self.source_combo.blockSignals(False)

        preview_values = ", ".join(str(v) for v in unique_values[:12])
        if len(unique_values) > 12:
            preview_values += f", ... ({len(unique_values)} total)"
        self.values_label.setText(f"Available values: [{preview_values}]")
        nonzero_values = [v for v in unique_values if v != 0]
        if nonzero_values:
            self.status_label.setText(
                f"[{layer.name}] contains {len(nonzero_values)} nonzero class value(s). "
                "Choose one source value and the target value."
            )
        else:
            self.status_label.setText(f"[{layer.name}] currently contains only background value 0.")

    def run_relabel(self):
        layer = self._current_layer()
        if layer is None:
            self.status_label.setText("No Labels layer is selected.")
            return
        source_text = str(self.source_combo.currentText() or "").strip()
        if not source_text:
            self.status_label.setText("Choose or type a source label value.")
            return
        try:
            source_value = int(source_text)
        except Exception:
            self.status_label.setText("Source value must be an integer.")
            return
        target_value = int(self.target_spin.value())
        prepared = prepare_tool_job(
            self.viewer,
            "replace_label_value",
            {
                "layer_name": layer.name,
                "source_value": source_value,
                "target_value": target_value,
            },
        )
        if prepared.get("mode") == "immediate":
            self.status_label.setText(str(prepared.get("message", "") or "Relabeling did not run."))
            self.refresh_from_viewer()
            return
        result = run_tool_job(prepared["job"])
        message = apply_tool_job_result(self.viewer, result)
        self.status_label.setText(str(message or "Relabeling completed."))
        self.refresh_from_viewer()


def open_relabel_mask_widget(viewer: napari.Viewer):
    existing_widget = getattr(viewer.window, "dock_widgets", {}).get(WIDGET_NAME)
    if existing_widget is not None:
        try:
            existing_dock = _dock_wrapper_from_widget(existing_widget)
            if existing_dock is None:
                raise RuntimeError("Dock wrapper not found.")
            existing_dock.setFloating(True)
            existing_dock.show()
            existing_dock.raise_()
            existing_widget.refresh_from_viewer()
            return existing_widget
        except Exception:
            pass

    widget = RelabelMaskWidget(viewer)
    dock_widget = viewer.window.add_dock_widget(widget, name=WIDGET_NAME, area="right")
    try:
        dock_widget.setFloating(True)
        dock_widget.show()
        dock_widget.raise_()
    except Exception:
        pass
    return widget
