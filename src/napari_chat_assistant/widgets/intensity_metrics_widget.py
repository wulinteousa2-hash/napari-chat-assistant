from __future__ import annotations

import csv
import io
from typing import Any

import napari
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (
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


WIDGET_NAME = "ROI Intensity Metrics"
PROMPT_OBJECT_NAME = "napariChatAssistantPrompt"
SHAPES_LAYER_NAME = "template_intensity_roi"


class IntensityMetricsWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.measurement_image_layer: napari.layers.Image | None = None
        self.shapes_layer = None
        self._rows: list[dict[str, Any]] = []
        self._refresh_pending = False

        self.setMinimumWidth(760)
        self.setWindowTitle(WIDGET_NAME)

        layout = QVBoxLayout(self)

        self.status_label = QLabel("Select an image layer and draw one or more ROI shapes to measure.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("Table View:"))
        self.table_view_combo = QComboBox()
        self.table_view_combo.addItems(["Absolute", "Percent"])
        view_row.addWidget(self.table_view_combo)
        view_row.addWidget(QLabel("Plot View:"))
        self.plot_view_combo = QComboBox()
        self.plot_view_combo.addItems(["Absolute", "Normalized"])
        view_row.addWidget(self.plot_view_combo)
        view_row.addStretch(1)
        layout.addLayout(view_row)

        self.figure = Figure(figsize=(6.4, 3.0), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)
        layout.addWidget(self.canvas, 1)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["ROI", "Pixels", "Mean", "Std", "Median", "Min", "Max", "Sum"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)

        button_row = QHBoxLayout()
        self.rename_btn = QPushButton("Rename ROI")
        self.copy_selected_btn = QPushButton("Copy Selected")
        self.copy_all_btn = QPushButton("Copy All")
        self.insert_chat_btn = QPushButton("Insert to Chat")
        self.export_csv_btn = QPushButton("Export CSV")
        button_row.addWidget(self.rename_btn)
        button_row.addWidget(self.copy_selected_btn)
        button_row.addWidget(self.copy_all_btn)
        button_row.addWidget(self.insert_chat_btn)
        button_row.addWidget(self.export_csv_btn)
        layout.addLayout(button_row)

        self.rename_btn.clicked.connect(self.rename_selected_roi)
        self.copy_selected_btn.clicked.connect(self.copy_selected_rows)
        self.copy_all_btn.clicked.connect(self.copy_all_rows)
        self.insert_chat_btn.clicked.connect(self.insert_rows_to_chat)
        self.export_csv_btn.clicked.connect(self.export_csv)
        self.table.itemSelectionChanged.connect(self._update_plot)
        self.table_view_combo.currentTextChanged.connect(self._refresh_table_view)
        self.plot_view_combo.currentTextChanged.connect(self._update_plot)

        self._ensure_shapes_layer()
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
            self.shapes_layer.events.data.connect(self._schedule_refresh)

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
                self.shapes_layer.events.data.disconnect(self._schedule_refresh)
            except Exception:
                pass

    def _schedule_refresh(self, event=None) -> None:
        del event
        if self._refresh_pending:
            return
        self._refresh_pending = True
        QTimer.singleShot(0, self._run_scheduled_refresh)

    def _run_scheduled_refresh(self) -> None:
        self._refresh_pending = False
        self.refresh()

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

    def _ensure_shapes_layer(self) -> None:
        existing = self._find_layer_by_name(SHAPES_LAYER_NAME)
        if isinstance(existing, napari.layers.Shapes):
            self.shapes_layer = existing
            self._sync_roi_labels()
            return

        self.shapes_layer = self.viewer.add_shapes(
            [self._default_roi_geometry()],
            shape_type="rectangle",
            edge_color="yellow",
            face_color="transparent",
            edge_width=2,
            name=SHAPES_LAYER_NAME,
            features={"label": np.asarray(["ROI 1"], dtype=object)},
        )
        self._sync_roi_labels()

    def _default_roi_label(self, index: int) -> str:
        return f"ROI {index + 1}"

    def _shape_count(self) -> int:
        try:
            return len(self.shapes_layer.data) if self.shapes_layer is not None else 0
        except Exception:
            return 0

    def _shape_labels(self) -> list[str]:
        if self.shapes_layer is None:
            return []
        features = getattr(self.shapes_layer, "features", None)
        labels: list[str] = []
        if features is not None and "label" in features:
            try:
                labels = [str(value).strip() for value in features["label"]]
            except Exception:
                labels = []
        return labels

    def _sync_roi_labels(self) -> None:
        if self.shapes_layer is None:
            return
        count = self._shape_count()
        labels = self._shape_labels()
        normalized = []
        for index in range(count):
            label = labels[index] if index < len(labels) else ""
            normalized.append(label or self._default_roi_label(index))
        try:
            self.shapes_layer.features = {"label": np.asarray(normalized, dtype=object)}
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

    def _pick_measurement_layer(self):
        active = self.viewer.layers.selection.active
        if self._is_live_image_layer(active):
            return active
        if self._is_live_image_layer(self.measurement_image_layer):
            return self.measurement_image_layer
        candidates = [layer for layer in self.viewer.layers if isinstance(layer, napari.layers.Image)]
        if candidates:
            return candidates[-1]
        return None

    def _get_image_data(self, layer: napari.layers.Image) -> tuple[np.ndarray, str]:
        data = np.asarray(layer.data, dtype=np.float32)
        if data.ndim < 2:
            raise ValueError(f"Layer '{layer.name}' must have at least 2 dimensions.")
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
        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = [
                row["roi"],
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
                self.table.setItem(row_index, column_index, item)
        self.table.resizeColumnsToContents()
        self._update_plot()

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

    def _update_plot(self) -> None:
        self.axes.clear()
        rows = self._selected_rows() or self._rows
        plotted = 0
        normalized = self.plot_view_combo.currentText() == "Normalized"
        for row in rows:
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
                bins = np.linspace(0.0, 100.0, 33)
                hist, edges = np.histogram(plot_values, bins=bins)
                scale = float(np.sum(hist))
                if scale > 0.0:
                    hist = (hist / scale) * 100.0
                centers = (edges[:-1] + edges[1:]) / 2.0
                self.axes.plot(centers, hist, linewidth=1.8, label=str(row["roi"]))
            else:
                bins = np.histogram_bin_edges(values, bins=32)
                hist, edges = np.histogram(values, bins=bins)
                centers = (edges[:-1] + edges[1:]) / 2.0
                self.axes.plot(centers, hist, linewidth=1.8, label=str(row["roi"]))
            plotted += 1

        if plotted:
            self.axes.set_title("ROI Intensity Histogram" if not normalized else "ROI Intensity Histogram (Normalized)")
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
        writer.writerow(["ROI", "Pixels", "Mean", "Std", "Median", "Min", "Max", "Sum"])
        for row in rows:
            writer.writerow(
                [
                    row["roi"],
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

    def rename_selected_roi(self) -> None:
        selected = sorted({index.row() for index in self.table.selectionModel().selectedRows()})
        if len(selected) != 1:
            self._set_status("Select exactly one ROI row to rename.")
            return
        row_index = selected[0]
        current_label = self._roi_label(row_index)
        new_label, accepted = QInputDialog.getText(self, "Rename ROI", "ROI name:", text=current_label)
        if not accepted:
            return
        new_label = str(new_label).strip() or self._default_roi_label(row_index)
        labels = self._shape_labels()
        count = self._shape_count()
        if len(labels) < count:
            labels.extend(self._default_roi_label(index) for index in range(len(labels), count))
        if row_index >= count:
            self._set_status("Selected ROI is no longer available.")
            return
        labels[row_index] = new_label
        try:
            self.shapes_layer.features = {"label": np.asarray(labels[:count], dtype=object)}
        except Exception:
            self._set_status("Could not rename the selected ROI.")
            return
        self.refresh()
        self._set_status(f"Renamed ROI to '{new_label}'.")

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
            "Export ROI Intensity Metrics",
            "roi_intensity_metrics.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["ROI", "Pixels", "Mean", "Std", "Median", "Min", "Max", "Sum"])
            for row in self._rows:
                writer.writerow(
                    [
                        row["roi"],
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

    def refresh(self, event=None) -> None:
        del event
        if self.shapes_layer is None or self.shapes_layer not in list(self.viewer.layers):
            existing = self._find_layer_by_name(SHAPES_LAYER_NAME)
            self.shapes_layer = existing if isinstance(existing, napari.layers.Shapes) else None
            if self.shapes_layer is not None:
                try:
                    self.shapes_layer.events.data.connect(self._schedule_refresh)
                except Exception:
                    pass
        self._sync_roi_labels()

        if self.shapes_layer is None:
            self._set_table_rows([])
            self._set_status("ROI layer is missing. Reopen ROI Intensity Metrics to recreate the helper ROI layer.")
            return

        self.measurement_image_layer = self._pick_measurement_layer()
        if self.measurement_image_layer is None:
            self._set_table_rows([])
            self._set_status("Select an image layer to start ROI measurement.")
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
        for index, mask in enumerate(masks):
            roi_label = self._roi_label(index)
            mask_array = np.asarray(mask, dtype=bool)
            if mask_array.shape != image.shape:
                rows.append(
                    {
                        "roi": roi_label,
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
            f"Measuring layer '{self.measurement_image_layer.name}' ({plane_label}) with {len(rows)} ROI shape(s)."
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
            if getattr(existing_widget, "shapes_layer", None) is None or getattr(existing_widget, "shapes_layer", None) not in list(viewer.layers):
                existing_widget._ensure_shapes_layer()
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
