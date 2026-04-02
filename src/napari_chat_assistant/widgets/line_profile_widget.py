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
from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit


WIDGET_NAME = "Line Profile Gaussian Fit"
PROMPT_OBJECT_NAME = "napariChatAssistantPrompt"
SHAPES_LAYER_NAME = "template_profile_line"


def line_profile(image: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> tuple[np.ndarray, np.ndarray]:
    length = np.hypot(x1 - x0, y1 - y0)
    n_samples = max(int(length) + 1, 2)
    x = np.linspace(x0, x1, n_samples)
    y = np.linspace(y0, y1, n_samples)
    profile = map_coordinates(image, [y, x], order=1)
    distance = np.linspace(0.0, length, n_samples)
    return distance, profile


def gaussian(x: np.ndarray, baseline: float, amplitude: float, mean: float, sigma: float) -> np.ndarray:
    return baseline + amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma**2))


class LineProfileGaussianFitWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.measurement_image_layer: napari.layers.Image | None = None
        self.shapes_layer = None
        self._rows: list[dict[str, Any]] = []
        self._refresh_pending = False

        self.setMinimumWidth(640)
        self.setWindowTitle(WIDGET_NAME)

        layout = QVBoxLayout(self)

        self.status_label = QLabel("Select an image layer to start measurement.")
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

        self.figure = Figure(figsize=(6.0, 3.0), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)
        layout.addWidget(self.canvas, 1)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Line", "Baseline", "Amplitude", "Mean", "Sigma", "FWHM", "Status"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)

        button_row = QHBoxLayout()
        self.rename_btn = QPushButton("Rename Line")
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

        self.rename_btn.clicked.connect(self.rename_selected_line)
        self.copy_selected_btn.clicked.connect(self.copy_selected_rows)
        self.copy_all_btn.clicked.connect(self.copy_all_rows)
        self.insert_chat_btn.clicked.connect(self.insert_rows_to_chat)
        self.export_csv_btn.clicked.connect(self.export_csv)
        self.table_view_combo.currentTextChanged.connect(self._refresh_table_view)
        self.plot_view_combo.currentTextChanged.connect(self._update_plot)
        self.table.itemSelectionChanged.connect(self._update_plot)

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

    def _ensure_shapes_layer(self) -> None:
        existing = self._find_layer_by_name(SHAPES_LAYER_NAME)
        if isinstance(existing, napari.layers.Shapes):
            self.shapes_layer = existing
            self._sync_line_labels()
            return

        self.shapes_layer = self.viewer.add_shapes(
            [[[96.0, 32.0], [32.0, 96.0]]],
            shape_type="line",
            edge_color="red",
            edge_width=3,
            name=SHAPES_LAYER_NAME,
            features={"label": np.asarray(["Line 1"], dtype=object)},
        )
        self._sync_line_labels()

    def _default_line_label(self, index: int) -> str:
        return f"Line {index + 1}"

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

    def _sync_line_labels(self) -> None:
        if self.shapes_layer is None:
            return
        count = self._shape_count()
        labels = self._shape_labels()
        normalized = []
        for index in range(count):
            label = labels[index] if index < len(labels) else ""
            normalized.append(label or self._default_line_label(index))
        try:
            self.shapes_layer.features = {"label": np.asarray(normalized, dtype=object)}
        except Exception:
            pass

    def _line_label(self, index: int) -> str:
        labels = self._shape_labels()
        if index < len(labels) and labels[index]:
            return labels[index]
        return self._default_line_label(index)

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

    def _clamp_point(self, x: float, y: float, image_shape: tuple[int, int]) -> tuple[float, float]:
        y_size, x_size = image_shape
        clamped_y = float(np.clip(y, 0, y_size - 1))
        clamped_x = float(np.clip(x, 0, x_size - 1))
        return clamped_y, clamped_x

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _set_table_rows(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = [
                row["line"],
                self._display_metric(row, "baseline"),
                self._display_metric(row, "amplitude"),
                self._display_metric(row, "mean"),
                self._display_metric(row, "sigma"),
                self._display_metric(row, "fwhm"),
                row["status"],
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

    def _display_metric(self, row: dict[str, Any], metric: str) -> str:
        value = row.get(f"{metric}_value")
        percent_mode = self.table_view_combo.currentText() == "Percent"
        if percent_mode:
            if metric in {"baseline", "amplitude"}:
                value = self._as_percent(value, row.get("profile_peak_value"))
            elif metric in {"mean", "sigma", "fwhm"}:
                value = self._as_percent(value, row.get("line_length_value"))
        return self._format_numeric(value, suffix="%" if percent_mode else "")

    def _update_plot(self) -> None:
        self.axes.clear()
        rows = self._selected_rows() or self._rows
        plotted = 0
        normalized = self.plot_view_combo.currentText() == "Normalized"
        for row in rows:
            distance = row.get("distance")
            profile = row.get("profile")
            fit = row.get("fit")
            if distance is None or profile is None:
                continue
            plot_profile = np.asarray(profile, dtype=float)
            plot_fit = None if fit is None else np.asarray(fit, dtype=float)
            if normalized:
                scale = float(np.max(np.abs(plot_profile))) if plot_profile.size else 0.0
                if scale > 0.0:
                    plot_profile = (plot_profile / scale) * 100.0
                    if plot_fit is not None:
                        plot_fit = (plot_fit / scale) * 100.0
            line_label = str(row["line"])
            self.axes.plot(distance, plot_profile, linewidth=1.8, label=f"{line_label} data")
            if plot_fit is not None:
                self.axes.plot(distance, plot_fit, linestyle="--", linewidth=1.4, label=f"{line_label} fit")
            plotted += 1
        if plotted:
            self.axes.set_title("Line Profile and Gaussian Fit" if not normalized else "Line Profile and Gaussian Fit (Normalized)")
            self.axes.set_xlabel("Distance (px)")
            self.axes.set_ylabel("Intensity" if not normalized else "Intensity (%)")
            self.axes.grid(True, alpha=0.25)
            self.axes.legend(loc="best", fontsize=8)
        else:
            self.axes.set_title("Line Profile and Gaussian Fit")
            self.axes.text(
                0.5,
                0.5,
                "No plottable line profile yet.",
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
        writer.writerow(["Line", "Baseline", "Amplitude", "Mean", "Sigma", "FWHM", "Status"])
        for row in rows:
            writer.writerow(
                [
                    row["line"],
                    self._display_metric(row, "baseline"),
                    self._display_metric(row, "amplitude"),
                    self._display_metric(row, "mean"),
                    self._display_metric(row, "sigma"),
                    self._display_metric(row, "fwhm"),
                    row["status"],
                ]
            )
        return output.getvalue().strip()

    def _selected_rows(self) -> list[dict[str, Any]]:
        selected = sorted({index.row() for index in self.table.selectionModel().selectedRows()})
        return [self._rows[index] for index in selected if 0 <= index < len(self._rows)]

    def rename_selected_line(self) -> None:
        selected = sorted({index.row() for index in self.table.selectionModel().selectedRows()})
        if len(selected) != 1:
            self._set_status("Select exactly one line row to rename.")
            return
        row_index = selected[0]
        current_label = self._line_label(row_index)
        new_label, accepted = QInputDialog.getText(self, "Rename Line", "Line name:", text=current_label)
        if not accepted:
            return
        new_label = str(new_label).strip() or self._default_line_label(row_index)
        labels = self._shape_labels()
        count = self._shape_count()
        if len(labels) < count:
            labels.extend(self._default_line_label(index) for index in range(len(labels), count))
        if row_index >= count:
            self._set_status("Selected line is no longer available.")
            return
        labels[row_index] = new_label
        try:
            self.shapes_layer.features = {"label": np.asarray(labels[:count], dtype=object)}
        except Exception:
            self._set_status("Could not rename the selected line.")
            return
        self.refresh()
        self._set_status(f"Renamed line to '{new_label}'.")

    def copy_selected_rows(self) -> None:
        rows = self._selected_rows()
        if not rows:
            self._set_status("Select one or more result rows to copy.")
            return
        QApplication.clipboard().setText(self._format_rows_as_tsv(rows))
        self._set_status(f"Copied {len(rows)} selected measurement row(s).")

    def copy_all_rows(self) -> None:
        if not self._rows:
            self._set_status("No measurement rows are available to copy.")
            return
        QApplication.clipboard().setText(self._format_rows_as_tsv(self._rows))
        self._set_status(f"Copied {len(self._rows)} measurement row(s).")

    def insert_rows_to_chat(self) -> None:
        rows = self._selected_rows() or self._rows
        if not rows:
            self._set_status("No measurement rows are available to insert.")
            return
        prompt = _find_prompt_widget()
        if prompt is None:
            self._set_status("Chat prompt was not found. Use Copy instead.")
            return
        current = prompt.toPlainText().rstrip()
        content = self._format_rows_as_tsv(rows)
        prompt.setPlainText(f"{current}\n{content}" if current else content)
        prompt.setFocus()
        self._set_status(f"Inserted {len(rows)} measurement row(s) into chat.")

    def export_csv(self) -> None:
        if not self._rows:
            self._set_status("No measurement rows are available to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Line Profile Measurements", "line_profile_measurements.csv", "CSV Files (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Line", "Baseline", "Amplitude", "Mean", "Sigma", "FWHM", "Status"])
            for row in self._rows:
                writer.writerow(
                    [
                        row["line"],
                        self._display_metric(row, "baseline"),
                        self._display_metric(row, "amplitude"),
                        self._display_metric(row, "mean"),
                        self._display_metric(row, "sigma"),
                        self._display_metric(row, "fwhm"),
                        row["status"],
                    ]
                )
        self._set_status(f"Exported {len(self._rows)} measurement row(s) to {path}.")

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

        if self.shapes_layer is None:
            self._set_table_rows([])
            self._set_status("Line ROI layer is missing. Reopen Line Profile Gaussian Fit to recreate the helper line layer.")
            return

        self.measurement_image_layer = self._pick_measurement_layer()
        if self.measurement_image_layer is None:
            self._set_table_rows([])
            self._set_status("Select an image layer to start measurement.")
            return

        try:
            image, plane_label = self._get_image_data(self.measurement_image_layer)
        except Exception as exc:
            self._set_table_rows([])
            self._set_status(str(exc))
            return

        rows: list[dict[str, Any]] = []
        for index, raw_line in enumerate(self.shapes_layer.data, start=1):
            line_label = self._line_label(index - 1)
            line_data = np.asarray(raw_line)
            if line_data.shape != (2, 2):
                rows.append(
                    {
                        "line": line_label,
                        "baseline_value": None,
                        "amplitude_value": None,
                        "mean_value": None,
                        "sigma_value": None,
                        "fwhm_value": None,
                        "line_length_value": None,
                        "profile_peak_value": None,
                        "status": "Skipped: not a straight line",
                        "distance": None,
                        "profile": None,
                        "fit": None,
                    }
                )
                continue

            (y0, x0), (y1, x1) = line_data
            y0, x0 = self._clamp_point(float(x0), float(y0), image.shape)
            y1, x1 = self._clamp_point(float(x1), float(y1), image.shape)
            dist, prof = line_profile(image, x0, y0, x1, y1)
            p0 = [
                float(np.min(prof)),
                float(np.max(prof) - np.min(prof)),
                float(dist[np.argmax(prof)]),
                5.0,
            ]
            try:
                params, _ = curve_fit(gaussian, dist, prof, p0=p0, maxfev=5000)
                baseline, amplitude, mean, sigma = [float(value) for value in params]
                fwhm = 2.3548 * sigma
                rows.append(
                    {
                        "line": line_label,
                        "baseline_value": baseline,
                        "amplitude_value": amplitude,
                        "mean_value": mean,
                        "sigma_value": sigma,
                        "fwhm_value": fwhm,
                        "line_length_value": float(dist[-1]) if len(dist) else None,
                        "profile_peak_value": float(np.max(prof)) if len(prof) else None,
                        "status": "OK",
                        "distance": dist,
                        "profile": prof,
                        "fit": gaussian(dist, baseline, amplitude, mean, sigma),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "line": line_label,
                        "baseline_value": None,
                        "amplitude_value": None,
                        "mean_value": None,
                        "sigma_value": None,
                        "fwhm_value": None,
                        "line_length_value": float(dist[-1]) if len(dist) else None,
                        "profile_peak_value": float(np.max(prof)) if len(prof) else None,
                        "status": f"Fit failed: {exc}",
                        "distance": dist,
                        "profile": prof,
                        "fit": None,
                    }
                )

        self._set_table_rows(rows)
        self._set_status(
            f"Measuring layer '{self.measurement_image_layer.name}' ({plane_label}) with {len(rows)} line(s)."
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


def open_line_profile_gaussian_fit_widget(viewer: napari.Viewer):
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

    widget = LineProfileGaussianFitWidget(viewer)
    dock_widget = viewer.window.add_dock_widget(widget, name=WIDGET_NAME, area="right")
    try:
        dock_widget.setFloating(True)
        dock_widget.show()
        dock_widget.raise_()
    except Exception:
        pass
    return widget
