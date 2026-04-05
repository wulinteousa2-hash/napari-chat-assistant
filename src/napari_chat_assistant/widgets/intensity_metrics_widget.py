from __future__ import annotations

import csv
import io
from typing import Any

import napari
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import QTimer, Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


WIDGET_NAME = "ROI Intensity Analysis"
PROMPT_OBJECT_NAME = "napariChatAssistantPrompt"
PAIR_LAYER_SUFFIX = "_intensity_roi"


class IntensityMetricsWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.measurement_image_layer: napari.layers.Image | None = None
        self.shapes_layer = None
        self._rows: list[dict[str, Any]] = []
        self._refresh_pending = False
        self._refresh_apply_visibility = True
        self._refresh_requeue = False
        self._updating_table = False
        self._hidden_shape_indices: set[int] = set()

        self.setMinimumWidth(760)
        self.setWindowTitle(WIDGET_NAME)

        layout = QVBoxLayout(self)

        self.status_label = QLabel("Select an image layer and draw one or more ROI shapes to measure.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        image_row = QHBoxLayout()
        image_row.addWidget(QLabel("Image Layer:"))
        self.image_layer_combo = QComboBox()
        image_row.addWidget(self.image_layer_combo, 1)
        self.create_pair_btn = QPushButton("Create Pair")
        self.isolate_pair_checkbox = QComboBox()
        self.isolate_pair_checkbox.addItems(["Show Only Active Pair", "Keep Other Layers Visible"])
        image_row.addWidget(self.create_pair_btn)
        image_row.addWidget(self.isolate_pair_checkbox)
        layout.addLayout(image_row)

        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("Table View:"))
        self.table_view_combo = QComboBox()
        self.table_view_combo.addItems(["Absolute", "Percent"])
        view_row.addWidget(self.table_view_combo)
        view_row.addWidget(QLabel("Plot View:"))
        self.plot_view_combo = QComboBox()
        self.plot_view_combo.addItems(["Absolute", "Normalized"])
        view_row.addWidget(self.plot_view_combo)
        view_row.addWidget(QLabel("Plot Style:"))
        self.plot_style_combo = QComboBox()
        self.plot_style_combo.addItems(["Line", "Bar"])
        view_row.addWidget(self.plot_style_combo)
        view_row.addWidget(QLabel("Bins:"))
        self.plot_bins_combo = QComboBox()
        self.plot_bins_combo.addItems(["16", "32", "64"])
        self.plot_bins_combo.setCurrentText("32")
        view_row.addWidget(self.plot_bins_combo)
        view_row.addStretch(1)
        layout.addLayout(view_row)

        self.figure = Figure(figsize=(6.4, 3.0), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)
        layout.addWidget(self.canvas, 1)

        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(["ROI", "State", "Pixels", "Mean", "Std", "Median", "Min", "Max", "Sum"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)

        button_row = QHBoxLayout()
        self.rename_btn = QPushButton("Rename ROI")
        self.hide_selected_btn = QPushButton("Hide Selected")
        self.show_selected_btn = QPushButton("Show Selected")
        self.remove_selected_btn = QPushButton("Remove Selected")
        self.copy_selected_btn = QPushButton("Copy Selected")
        self.copy_all_btn = QPushButton("Copy All")
        self.insert_chat_btn = QPushButton("Insert to Chat")
        self.export_csv_btn = QPushButton("Export CSV")
        button_row.addWidget(self.rename_btn)
        button_row.addWidget(self.hide_selected_btn)
        button_row.addWidget(self.show_selected_btn)
        button_row.addWidget(self.remove_selected_btn)
        button_row.addWidget(self.copy_selected_btn)
        button_row.addWidget(self.copy_all_btn)
        button_row.addWidget(self.insert_chat_btn)
        button_row.addWidget(self.export_csv_btn)
        layout.addLayout(button_row)

        self.rename_btn.clicked.connect(self.rename_selected_roi)
        self.hide_selected_btn.clicked.connect(self.hide_selected_rois)
        self.show_selected_btn.clicked.connect(self.show_selected_rois)
        self.remove_selected_btn.clicked.connect(self.remove_selected_rois)
        self.copy_selected_btn.clicked.connect(self.copy_selected_rows)
        self.copy_all_btn.clicked.connect(self.copy_all_rows)
        self.insert_chat_btn.clicked.connect(self.insert_rows_to_chat)
        self.export_csv_btn.clicked.connect(self.export_csv)
        self.table.itemSelectionChanged.connect(self._update_plot)
        self.table.itemChanged.connect(self._handle_table_item_changed)
        self.image_layer_combo.currentTextChanged.connect(self._schedule_refresh)
        self.create_pair_btn.clicked.connect(self.create_pair_for_current_image)
        self.isolate_pair_checkbox.currentTextChanged.connect(self._schedule_refresh)
        self.table_view_combo.currentTextChanged.connect(self._refresh_table_view)
        self.plot_view_combo.currentTextChanged.connect(self._update_plot)
        self.plot_style_combo.currentTextChanged.connect(self._update_plot)
        self.plot_bins_combo.currentTextChanged.connect(self._update_plot)

        self._connect_events()
        self.refresh()

    def _connect_events(self) -> None:
        self.viewer.layers.selection.events.active.connect(self._schedule_refresh)
        self.viewer.layers.selection.events.changed.connect(self._schedule_refresh)
        self.viewer.layers.events.inserted.connect(self._schedule_refresh)
        self.viewer.layers.events.removed.connect(self._schedule_refresh)
        self.viewer.layers.events.reordered.connect(self._schedule_refresh)
        self.viewer.dims.events.current_step.connect(self._schedule_refresh)
        if self.shapes_layer is not None:
            self.shapes_layer.events.data.connect(self._schedule_measurement_refresh)

    def _disconnect_events(self) -> None:
        disconnects = [
            (self.viewer.layers.selection.events.active, self._schedule_refresh),
            (self.viewer.layers.selection.events.changed, self._schedule_refresh),
            (self.viewer.layers.events.inserted, self._schedule_refresh),
            (self.viewer.layers.events.removed, self._schedule_refresh),
            (self.viewer.layers.events.reordered, self._schedule_refresh),
            (self.viewer.dims.events.current_step, self._schedule_refresh),
        ]
        for emitter, callback in disconnects:
            try:
                emitter.disconnect(callback)
            except Exception:
                pass
        if self.shapes_layer is not None:
            try:
                self.shapes_layer.events.data.disconnect(self._schedule_measurement_refresh)
            except Exception:
                pass

    def _schedule_refresh(self, event=None) -> None:
        del event
        if self._refresh_pending:
            self._refresh_apply_visibility = True
            self._refresh_requeue = True
            return
        self._refresh_pending = True
        self._refresh_apply_visibility = True
        QTimer.singleShot(0, self._run_scheduled_refresh)

    def _schedule_measurement_refresh(self, event=None) -> None:
        del event
        if self._refresh_pending:
            self._refresh_requeue = True
            return
        self._refresh_pending = True
        self._refresh_apply_visibility = False
        QTimer.singleShot(0, self._run_scheduled_refresh)

    def _run_scheduled_refresh(self) -> None:
        apply_visibility = self._refresh_apply_visibility
        requeue = self._refresh_requeue
        self._refresh_pending = False
        self._refresh_apply_visibility = True
        self._refresh_requeue = False
        self.refresh(apply_visibility=apply_visibility)
        if requeue:
            self._refresh_pending = True
            self._refresh_apply_visibility = True
            QTimer.singleShot(0, self._run_scheduled_refresh)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._disconnect_events()
        super().closeEvent(event)

    def _find_layer_by_name(self, name: str):
        for layer in self.viewer.layers:
            if getattr(layer, "name", None) == name:
                return layer
        return None

    def _default_roi_geometry(self) -> list[list[float]]:
        layer = self._pick_measurement_layer()
        if layer is None:
            return [[16.0, 16.0], [16.0, 96.0], [96.0, 96.0], [96.0, 16.0]]
        image, _ = self._get_image_data(layer)
        y_size, x_size = image.shape
        top = max(0.0, y_size * 0.2)
        left = max(0.0, x_size * 0.2)
        bottom = min(float(y_size - 1), y_size * 0.6)
        right = min(float(x_size - 1), x_size * 0.6)
        return [[top, left], [top, right], [bottom, right], [bottom, left]]

    def _pair_layer_name(self, image_layer_name: str) -> str:
        return f"{str(image_layer_name).strip()}{PAIR_LAYER_SUFFIX}"

    def _find_paired_shapes_layer(self, image_layer_name: str):
        target = str(image_layer_name or "").strip()
        if not target:
            return None
        named = self._find_layer_by_name(self._pair_layer_name(target))
        if isinstance(named, napari.layers.Shapes):
            return named
        for layer in self.viewer.layers:
            if not isinstance(layer, napari.layers.Shapes):
                continue
            metadata = dict(getattr(layer, "metadata", {}) or {})
            if str(metadata.get("paired_image_layer", "")).strip() == target:
                return layer
        return None

    def _attach_shapes_layer(self, layer) -> None:
        if layer is self.shapes_layer:
            return
        if self.shapes_layer is not None:
            try:
                self.shapes_layer.events.data.disconnect(self._schedule_measurement_refresh)
            except Exception:
                pass
        self.shapes_layer = layer if isinstance(layer, napari.layers.Shapes) else None
        if self.shapes_layer is not None:
            try:
                self.shapes_layer.events.data.connect(self._schedule_measurement_refresh)
            except Exception:
                pass
            self._sync_roi_labels()
        self._hidden_shape_indices = set()
        self._apply_hidden_state()

    def _default_roi_label(self, index: int) -> str:
        return f"ROI {index + 1}"

    def _normalize_roi_label_value(self, value: Any, index: int) -> str:
        text = str(value).strip() if value is not None else ""
        if not text or text.lower() in {"none", "nan"}:
            return self._default_roi_label(index)
        return text

    def _shape_count(self) -> int:
        try:
            return len(self.shapes_layer.data) if self.shapes_layer is not None else 0
        except Exception:
            return 0

    def _shape_types(self, layer=None) -> list[str]:
        target = layer if layer is not None else self.shapes_layer
        if target is None:
            return []
        try:
            raw = getattr(target, "shape_type", [])
            if isinstance(raw, str):
                return [raw]
            return [str(value).strip().lower() for value in list(raw)]
        except Exception:
            return []

    def _area_shape_indices(self, layer=None) -> list[int]:
        area_types = {"rectangle", "ellipse", "polygon"}
        return [index for index, shape_type in enumerate(self._shape_types(layer)) if shape_type in area_types]

    def _is_intensity_pair_layer(self, layer) -> bool:
        if not isinstance(layer, napari.layers.Shapes):
            return False
        name = str(getattr(layer, "name", "") or "").strip()
        metadata = dict(getattr(layer, "metadata", {}) or {})
        return name.endswith(PAIR_LAYER_SUFFIX) or bool(str(metadata.get("paired_image_layer", "")).strip())

    def _shape_labels(self) -> list[str]:
        if self.shapes_layer is None:
            return []
        features = getattr(self.shapes_layer, "features", None)
        labels: list[str] = []
        if features is not None and "label" in features:
            try:
                labels = [
                    self._normalize_roi_label_value(value, index)
                    for index, value in enumerate(features["label"])
                ]
            except Exception:
                labels = []
        return labels

    def _apply_shape_text_labels(self, labels: list[str]) -> None:
        if self.shapes_layer is None:
            return
        try:
            self.shapes_layer.text = {
                "string": "{label}",
                "size": 10,
                "color": "yellow",
                "anchor": "upper_left",
                "translation": np.array([0.0, -6.0]),
            }
        except Exception:
            return

    def _sync_roi_labels(self) -> None:
        if self.shapes_layer is None:
            return
        count = self._shape_count()
        labels = self._shape_labels()
        normalized = []
        seen_labels: set[str] = set()
        for index in range(count):
            label = labels[index] if index < len(labels) else None
            normalized_label = self._normalize_roi_label_value(label, index)
            default_label = self._default_roi_label(index)
            if normalized_label in seen_labels and normalized_label.startswith("ROI "):
                normalized_label = default_label
            seen_labels.add(normalized_label)
            normalized.append(normalized_label)
        try:
            self.shapes_layer.features = {"label": np.asarray(normalized, dtype=object)}
            self._apply_shape_text_labels(normalized)
        except Exception:
            pass

    def _normalize_hidden_indices(self) -> None:
        count = self._shape_count()
        self._hidden_shape_indices = {index for index in self._hidden_shape_indices if 0 <= index < count}

    def _apply_hidden_state(self) -> None:
        if self.shapes_layer is None:
            return
        self._normalize_hidden_indices()
        shown = np.ones(self._shape_count(), dtype=bool)
        for index in self._hidden_shape_indices:
            if 0 <= index < len(shown):
                shown[index] = False
        try:
            self.shapes_layer.shown = shown
        except Exception:
            pass

    def _roi_label(self, index: int) -> str:
        labels = self._shape_labels()
        if index < len(labels) and labels[index]:
            return labels[index]
        return self._default_roi_label(index)

    def _is_live_image_layer(self, layer) -> bool:
        if layer is None or not isinstance(layer, napari.layers.Image):
            return False
        try:
            return layer in list(self.viewer.layers)
        except Exception:
            return False

    def _is_live_shapes_layer(self, layer) -> bool:
        if layer is None or not isinstance(layer, napari.layers.Shapes):
            return False
        try:
            return layer in list(self.viewer.layers)
        except Exception:
            return False

    def _image_layer_choices(self) -> list[napari.layers.Image]:
        return [layer for layer in self.viewer.layers if isinstance(layer, napari.layers.Image)]

    def _preferred_measurement_layer(self):
        active = self.viewer.layers.selection.active
        if self._is_live_image_layer(active):
            return active
        if self._is_live_image_layer(self.measurement_image_layer):
            return self.measurement_image_layer
        candidates = self._image_layer_choices()
        if candidates:
            return candidates[-1]
        return None

    def _refresh_image_layer_combo(self) -> None:
        current_text = self.image_layer_combo.currentText().strip()
        preferred = self._preferred_measurement_layer()
        names = [layer.name for layer in self._image_layer_choices()]
        self.image_layer_combo.blockSignals(True)
        self.image_layer_combo.clear()
        self.image_layer_combo.addItems(names)
        if current_text and current_text in names:
            self.image_layer_combo.setCurrentText(current_text)
        elif preferred is not None and preferred.name in names:
            self.image_layer_combo.setCurrentText(preferred.name)
        self.image_layer_combo.blockSignals(False)

    def _pick_measurement_layer(self):
        combo_name = self.image_layer_combo.currentText().strip()
        if combo_name:
            layer = self._find_layer_by_name(combo_name)
            if self._is_live_image_layer(layer):
                return layer
        return self._preferred_measurement_layer()

    def _preferred_shapes_layer(self, image_layer_name: str = ""):
        active = self.viewer.layers.selection.active
        if self._is_live_shapes_layer(active) and self._area_shape_indices(active):
            return active
        if image_layer_name:
            paired_active = self._find_paired_shapes_layer(image_layer_name)
            if self._is_live_shapes_layer(active) and active is paired_active:
                return active
        if self._is_live_shapes_layer(self.shapes_layer) and self._area_shape_indices(self.shapes_layer):
            return self.shapes_layer
        if image_layer_name:
            paired_current = self._find_paired_shapes_layer(image_layer_name)
            if self._is_live_shapes_layer(self.shapes_layer) and self.shapes_layer is paired_current:
                return self.shapes_layer
        if image_layer_name:
            paired = self._find_paired_shapes_layer(image_layer_name)
            if self._is_live_shapes_layer(paired):
                return paired
        return None

    def _show_only_active_pair(self) -> bool:
        return self.isolate_pair_checkbox.currentIndex() == 0

    def _apply_pair_visibility(self) -> None:
        image_layer = self.measurement_image_layer
        shapes_layer = self.shapes_layer
        if not self._show_only_active_pair() or image_layer is None or shapes_layer is None:
            return
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                layer.visible = layer is image_layer
            elif self._is_intensity_pair_layer(layer):
                layer.visible = layer is shapes_layer

    def create_pair_for_current_image(self) -> None:
        image_layer = self._pick_measurement_layer()
        if image_layer is None:
            self._set_status("Select an image layer first.")
            return
        existing = self._find_paired_shapes_layer(image_layer.name)
        if isinstance(existing, napari.layers.Shapes):
            self._attach_shapes_layer(existing)
            self.measurement_image_layer = image_layer
            self._apply_pair_visibility()
            self.refresh()
            self._set_status(f"Using existing ROI pair [{existing.name}] for [{image_layer.name}].")
            return

        shapes = self.viewer.add_shapes(
            data=[],
            shape_type=[],
            edge_color="yellow",
            face_color="transparent",
            edge_width=2,
            name=self._pair_layer_name(image_layer.name),
            features={"label": np.asarray([], dtype=object)},
            metadata={"paired_image_layer": image_layer.name},
        )
        self._attach_shapes_layer(shapes)
        self.measurement_image_layer = image_layer
        self._apply_pair_visibility()
        self.refresh()
        self._set_status(f"Created ROI pair [{shapes.name}] for [{image_layer.name}]. Draw one or more shapes to measure.")

    def _get_image_data(self, layer: napari.layers.Image) -> tuple[np.ndarray, str]:
        data = np.asarray(layer.data, dtype=np.float32)
        if data.ndim < 2:
            raise ValueError(f"Layer '{layer.name}' must have at least 2 dimensions.")
        if bool(getattr(layer, "rgb", False)) and data.ndim >= 3:
            channels = min(int(data.shape[-1]), 3)
            if channels <= 0:
                raise ValueError(f"Layer '{layer.name}' does not contain usable RGB channels.")
            if channels == 1:
                plane = data[..., 0]
            else:
                weights = np.asarray([0.2126, 0.7152, 0.0722][:channels], dtype=np.float32)
                weights = weights / float(np.sum(weights))
                plane = np.tensordot(data[..., :channels], weights, axes=([-1], [0]))
            return np.asarray(plane, dtype=np.float32), "full image RGB luminance"
        if data.ndim == 2:
            return data, "full image"

        current_step = tuple(int(step) for step in self.viewer.dims.current_step[: data.ndim])
        leading_shape = data.shape[:-2]
        leading_indices = []
        for axis, axis_size in enumerate(leading_shape):
            step_index = current_step[axis] if axis < len(current_step) else 0
            leading_indices.append(int(np.clip(step_index, 0, axis_size - 1)))

        plane = data[tuple(leading_indices) + (slice(None), slice(None))]
        plane_label = ", ".join(f"axis {axis}={index}" for axis, index in enumerate(leading_indices, start=1))
        return np.asarray(plane, dtype=np.float32), plane_label

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _image_context(self, image: np.ndarray) -> dict[str, float | None]:
        finite = image[np.isfinite(image)]
        if finite.size == 0:
            return {
                "image_min": None,
                "image_max": None,
                "image_range": None,
                "total_pixels": float(image.size),
                "total_sum": None,
            }
        image_min = float(np.min(finite))
        image_max = float(np.max(finite))
        return {
            "image_min": image_min,
            "image_max": image_max,
            "image_range": float(image_max - image_min),
            "total_pixels": float(image.size),
            "total_sum": float(np.sum(finite)),
        }

    def _set_table_rows(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self._updating_table = True
        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = [
                row["roi"],
                row.get("state", "Shown"),
                self._display_metric(row, "pixels"),
                self._display_metric(row, "mean"),
                self._display_metric(row, "std"),
                self._display_metric(row, "median"),
                self._display_metric(row, "min"),
                self._display_metric(row, "max"),
                self._display_metric(row, "sum"),
            ]
            for column_index, value in enumerate(values):
                item = QTableWidgetItem("" if value is None else str(value))
                if column_index == 0:
                    item.setFlags(item.flags() | Qt.ItemIsEditable)
                else:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row_index, column_index, item)
        self._updating_table = False
        self.table.resizeColumnsToContents()
        self._update_plot()

    def _set_roi_label(self, row_index: int, new_label: str) -> bool:
        if self.shapes_layer is None:
            self._set_status("No ROI layer is attached.")
            return False
        count = self._shape_count()
        if row_index < 0 or row_index >= count:
            self._set_status("Selected ROI is no longer available.")
            return False
        labels = self._shape_labels()
        if len(labels) < count:
            labels.extend(self._default_roi_label(index) for index in range(len(labels), count))
        labels[row_index] = str(new_label).strip() or self._default_roi_label(row_index)
        try:
            self.shapes_layer.features = {"label": np.asarray(labels[:count], dtype=object)}
            self._apply_shape_text_labels(labels[:count])
        except Exception:
            self._set_status("Could not rename the selected ROI.")
            return False
        self.refresh(apply_visibility=False)
        self._set_status(f"Renamed ROI to '{labels[row_index]}'.")
        return True

    def _handle_table_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_table or item is None:
            return
        if item.column() != 0:
            return
        self._set_roi_label(item.row(), item.text())

    def _refresh_table_view(self) -> None:
        self._set_table_rows(self._rows)

    def _as_percent(self, numerator: float | None, denominator: float | None) -> float | None:
        if numerator is None or denominator is None or abs(denominator) < 1e-12:
            return None
        return (numerator / denominator) * 100.0

    def _range_percent(self, value: float | None, image_min: float | None, image_range: float | None) -> float | None:
        if value is None or image_min is None or image_range is None or abs(image_range) < 1e-12:
            return None
        return ((value - image_min) / image_range) * 100.0

    def _display_metric(self, row: dict[str, Any], metric: str) -> str:
        value = row.get(f"{metric}_value")
        percent_mode = self.table_view_combo.currentText() == "Percent"
        if percent_mode:
            if metric == "pixels":
                value = self._as_percent(value, row.get("total_pixels"))
            elif metric == "sum":
                value = self._as_percent(value, row.get("total_sum"))
            elif metric == "std":
                value = self._as_percent(value, row.get("image_range"))
            else:
                value = self._range_percent(value, row.get("image_min"), row.get("image_range"))
        suffix = "%" if percent_mode else ""
        if metric == "pixels" and not percent_mode and value is not None:
            return str(int(round(float(value))))
        return self._format_numeric(value, suffix=suffix)

    def _plot_bin_count(self) -> int:
        try:
            return max(4, int(self.plot_bins_combo.currentText().strip() or "32"))
        except Exception:
            return 32

    def _update_plot(self) -> None:
        self.axes.clear()
        rows = self._selected_rows() or self._rows
        plotted = 0
        normalized = self.plot_view_combo.currentText() == "Normalized"
        plot_style = self.plot_style_combo.currentText()
        bin_count = self._plot_bin_count()
        for row in rows:
            if str(row.get("state", "")).strip().lower() == "hidden":
                continue
            values = row.get("values")
            if values is None:
                continue
            values = np.asarray(values, dtype=float)
            if values.size == 0:
                continue
            if normalized:
                image_min = row.get("image_min")
                image_range = row.get("image_range")
                if image_min is None or image_range is None or abs(image_range) < 1e-12:
                    continue
                plot_values = ((values - float(image_min)) / float(image_range)) * 100.0
                bins = np.linspace(0.0, 100.0, bin_count + 1)
                hist, edges = np.histogram(plot_values, bins=bins)
                scale = float(np.sum(hist))
                if scale > 0.0:
                    hist = (hist / scale) * 100.0
                centers = (edges[:-1] + edges[1:]) / 2.0
            else:
                bins = np.histogram_bin_edges(values, bins=bin_count)
                hist, edges = np.histogram(values, bins=bins)
                centers = (edges[:-1] + edges[1:]) / 2.0
            label = str(row["roi"])
            if plot_style == "Bar":
                widths = np.diff(edges)
                self.axes.bar(centers, hist, width=widths, alpha=0.35, align="center", label=label)
            else:
                self.axes.plot(centers, hist, linewidth=1.8, label=label)
            plotted += 1

        if plotted:
            style_suffix = " (Bar)" if plot_style == "Bar" else ""
            self.axes.set_title(
                ("ROI Intensity Histogram" if not normalized else "ROI Intensity Histogram (Normalized)") + style_suffix
            )
            self.axes.set_xlabel("Intensity" if not normalized else "Intensity (%)")
            self.axes.set_ylabel("Count" if not normalized else "Pixels (%)")
            self.axes.grid(True, alpha=0.25)
            self.axes.legend(loc="best", fontsize=8)
        else:
            self.axes.set_title("ROI Intensity Histogram")
            self.axes.text(
                0.5,
                0.5,
                "No plottable ROI histogram yet.",
                ha="center",
                va="center",
                transform=self.axes.transAxes,
            )
            self.axes.set_xticks([])
            self.axes.set_yticks([])
        self.canvas.draw_idle()

    def _format_numeric(self, value: float | None, suffix: str = "") -> str:
        if value is None:
            return ""
        return f"{value:.2f}{suffix}"

    def _format_rows_as_tsv(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return ""
        output = io.StringIO()
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow(["ROI", "State", "Pixels", "Mean", "Std", "Median", "Min", "Max", "Sum"])
        for row in rows:
            writer.writerow(
                [
                    row["roi"],
                    row.get("state", "Shown"),
                    self._display_metric(row, "pixels"),
                    self._display_metric(row, "mean"),
                    self._display_metric(row, "std"),
                    self._display_metric(row, "median"),
                    self._display_metric(row, "min"),
                    self._display_metric(row, "max"),
                    self._display_metric(row, "sum"),
                ]
            )
        return output.getvalue().strip()

    def _selected_rows(self) -> list[dict[str, Any]]:
        selected = sorted({index.row() for index in self.table.selectionModel().selectedRows()})
        return [self._rows[index] for index in selected if 0 <= index < len(self._rows)]

    def _selected_shape_indices(self) -> list[int]:
        indices: list[int] = []
        for row in self._selected_rows():
            shape_index = row.get("shape_index")
            if isinstance(shape_index, int):
                indices.append(shape_index)
        return sorted(set(indices))

    def rename_selected_roi(self) -> None:
        selected = sorted({index.row() for index in self.table.selectionModel().selectedRows()})
        if len(selected) != 1:
            self._set_status("Select exactly one ROI row to rename.")
            return
        shape_index = self._rows[selected[0]].get("shape_index")
        if not isinstance(shape_index, int):
            self._set_status("Selected ROI is no longer available.")
            return
        current_label = self._roi_label(shape_index)
        new_label, accepted = QInputDialog.getText(self, "Rename ROI", "ROI name:", text=current_label)
        if not accepted:
            return
        self._set_roi_label(shape_index, new_label)

    def hide_selected_rois(self) -> None:
        indices = self._selected_shape_indices()
        if not indices:
            self._set_status("Select one or more ROI rows to hide.")
            return
        self._hidden_shape_indices.update(indices)
        self._apply_hidden_state()
        self.refresh(apply_visibility=False)
        self._set_status(f"Hid {len(indices)} ROI shape(s).")

    def show_selected_rois(self) -> None:
        indices = self._selected_shape_indices()
        if not indices:
            self._set_status("Select one or more ROI rows to show.")
            return
        self._hidden_shape_indices.difference_update(indices)
        self._apply_hidden_state()
        self.refresh(apply_visibility=False)
        self._set_status(f"Showed {len(indices)} ROI shape(s).")

    def remove_selected_rois(self) -> None:
        indices = self._selected_shape_indices()
        if not indices:
            self._set_status("Select one or more ROI rows to remove.")
            return
        if self.shapes_layer is None:
            self._set_status("No ROI layer is attached.")
            return
        try:
            self.shapes_layer.selected_data = set(indices)
            self.shapes_layer.remove_selected()
        except Exception:
            self._set_status("Could not remove the selected ROI shape(s).")
            return
        removed = len(indices)
        self._hidden_shape_indices = set()
        self.refresh(apply_visibility=False)
        self._set_status(f"Removed {removed} ROI shape(s).")

    def copy_selected_rows(self) -> None:
        rows = self._selected_rows()
        if not rows:
            self._set_status("Select one or more ROI rows to copy.")
            return
        QApplication.clipboard().setText(self._format_rows_as_tsv(rows))
        self._set_status(f"Copied {len(rows)} selected ROI row(s).")

    def copy_all_rows(self) -> None:
        if not self._rows:
            self._set_status("No ROI rows are available to copy.")
            return
        QApplication.clipboard().setText(self._format_rows_as_tsv(self._rows))
        self._set_status(f"Copied {len(self._rows)} ROI row(s).")

    def insert_rows_to_chat(self) -> None:
        rows = self._selected_rows() or self._rows
        if not rows:
            self._set_status("No ROI rows are available to insert.")
            return
        prompt = _find_prompt_widget()
        if prompt is None:
            self._set_status("Chat prompt was not found. Use Copy instead.")
            return
        current = prompt.toPlainText().rstrip()
        content = self._format_rows_as_tsv(rows)
        prompt.setPlainText(f"{current}\n{content}" if current else content)
        prompt.setFocus()
        self._set_status(f"Inserted {len(rows)} ROI row(s) into chat.")

    def export_csv(self) -> None:
        if not self._rows:
            self._set_status("No ROI rows are available to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export ROI Intensity Analysis",
            "roi_intensity_metrics.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["ROI", "State", "Pixels", "Mean", "Std", "Median", "Min", "Max", "Sum"])
            for row in self._rows:
                writer.writerow(
                    [
                        row["roi"],
                        row.get("state", "Shown"),
                        self._display_metric(row, "pixels"),
                        self._display_metric(row, "mean"),
                        self._display_metric(row, "std"),
                        self._display_metric(row, "median"),
                        self._display_metric(row, "min"),
                        self._display_metric(row, "max"),
                        self._display_metric(row, "sum"),
                    ]
                )
        self._set_status(f"Exported {len(self._rows)} ROI row(s) to {path}.")

    def refresh(self, event=None, *, apply_visibility: bool = True) -> None:
        del event
        self._refresh_image_layer_combo()

        self.measurement_image_layer = self._pick_measurement_layer()
        if self.measurement_image_layer is None:
            self._attach_shapes_layer(None)
            self._set_table_rows([])
            self._set_status("Select an image layer to start ROI measurement.")
            return
        self._attach_shapes_layer(self._preferred_shapes_layer(self.measurement_image_layer.name))
        self._apply_hidden_state()
        if apply_visibility:
            self._apply_pair_visibility()

        if self.shapes_layer is None:
            self._set_table_rows([])
            self._set_status(
                f"Selected image [{self.measurement_image_layer.name}]. Select or draw a Shapes ROI, or click Create Pair for a dedicated ROI layer."
            )
            return
        self._sync_roi_labels()

        area_indices = self._area_shape_indices(self.shapes_layer)
        if not area_indices:
            self._set_table_rows([])
            self._set_status(
                f"Selected Shapes layer [{self.shapes_layer.name}] does not contain area ROIs. Use rectangles, ellipses, or polygons for ROI intensity analysis."
            )
            return

        try:
            image, plane_label = self._get_image_data(self.measurement_image_layer)
        except Exception as exc:
            self._set_table_rows([])
            self._set_status(str(exc))
            return

        context = self._image_context(image)
        rows: list[dict[str, Any]] = []
        masks = self.shapes_layer.to_masks(mask_shape=image.shape) if self.shapes_layer is not None else []
        for index in area_indices:
            if index >= len(masks):
                continue
            mask = masks[index]
            roi_label = self._roi_label(index)
            hidden = index in self._hidden_shape_indices
            mask_array = np.asarray(mask, dtype=bool)
            if mask_array.shape != image.shape:
                rows.append(
                    {
                        "roi": roi_label,
                        "state": "Hidden" if hidden else "Shown",
                        "shape_index": index,
                        "pixels_value": None,
                        "mean_value": None,
                        "std_value": None,
                        "median_value": None,
                        "min_value": None,
                        "max_value": None,
                        "sum_value": None,
                        "status": "Skipped: ROI mask shape mismatch",
                        "values": None,
                        **context,
                    }
                )
                continue
            values = image[mask_array]
            values = values[np.isfinite(values)]
            if values.size == 0:
                rows.append(
                    {
                        "roi": roi_label,
                        "state": "Hidden" if hidden else "Shown",
                        "shape_index": index,
                        "pixels_value": 0.0,
                        "mean_value": None,
                        "std_value": None,
                        "median_value": None,
                        "min_value": None,
                        "max_value": None,
                        "sum_value": None,
                        "status": "No finite pixels inside ROI",
                        "values": np.asarray([], dtype=float),
                        **context,
                    }
                )
                continue
            rows.append(
                {
                    "roi": roi_label,
                    "state": "Hidden" if hidden else "Shown",
                    "shape_index": index,
                    "pixels_value": float(values.size),
                    "mean_value": float(np.mean(values)),
                    "std_value": float(np.std(values)),
                    "median_value": float(np.median(values)),
                    "min_value": float(np.min(values)),
                    "max_value": float(np.max(values)),
                    "sum_value": float(np.sum(values)),
                    "status": "OK",
                    "values": values,
                    **context,
                }
            )

        self._set_table_rows(rows)
        self._set_status(
            f"Measuring layer '{self.measurement_image_layer.name}' with ROI layer '{self.shapes_layer.name}' ({plane_label}) and {len(rows)} ROI shape(s)."
        )


def _dock_wrapper_from_widget(widget):
    try:
        return widget.parent()
    except Exception:
        pass
    try:
        return widget.native.parent()
    except Exception:
        return None


def _find_prompt_widget():
    for widget in QApplication.allWidgets():
        try:
            if widget.objectName() == PROMPT_OBJECT_NAME and hasattr(widget, "toPlainText") and hasattr(widget, "setPlainText"):
                return widget
        except Exception:
            continue
    return None


def open_intensity_metrics_widget(viewer: napari.Viewer):
    existing_widget = getattr(viewer.window, "dock_widgets", {}).get(WIDGET_NAME)
    if existing_widget is not None:
        try:
            existing_dock = _dock_wrapper_from_widget(existing_widget)
            if existing_dock is None:
                raise RuntimeError("Dock wrapper not found.")
            existing_dock.setFloating(True)
            existing_dock.show()
            existing_dock.raise_()
            existing_widget.refresh()
            return existing_widget
        except Exception:
            pass

    widget = IntensityMetricsWidget(viewer)
    dock_widget = viewer.window.add_dock_widget(widget, name=WIDGET_NAME, area="right")
    try:
        dock_widget.setFloating(True)
        dock_widget.show()
        dock_widget.raise_()
    except Exception:
        pass
    return widget
