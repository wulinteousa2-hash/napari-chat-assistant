from __future__ import annotations

import csv
import io
from typing import Any

import napari
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from scipy import stats


WIDGET_NAME = "Group Comparison Stats"
PROMPT_OBJECT_NAME = "napariChatAssistantPrompt"


def _find_prompt_widget():
    for widget in QApplication.allWidgets():
        try:
            if widget.objectName() == PROMPT_OBJECT_NAME and hasattr(widget, "toPlainText") and hasattr(widget, "setPlainText"):
                return widget
        except Exception:
            continue
    return None


def _dock_wrapper_from_widget(widget):
    try:
        return widget.parent()
    except Exception:
        pass
    try:
        return widget.native.parent()
    except Exception:
        return None


class GroupComparisonWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self._rows: list[dict[str, Any]] = []
        self._report_lines: list[str] = []

        self.setMinimumWidth(840)
        self.setWindowTitle(WIDGET_NAME)

        layout = QVBoxLayout(self)

        self.status_label = QLabel("Compare two groups by prefix. Use whole-image mode or ROI-based mode.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        form = QFormLayout()
        self.group_a_edit = QLineEdit("wt_")
        self.group_b_edit = QLineEdit("mutant_")
        self.scope_combo = QComboBox()
        self.scope_combo.addItems(["Whole Image", "ROI-based"])
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["mean", "median", "sum", "std", "area"])
        self.roi_kind_combo = QComboBox()
        self.roi_kind_combo.addItems(["auto", "shapes", "labels"])
        self.pair_mode_combo = QComboBox()
        self.pair_mode_combo.addItems(["unpaired", "paired_suffix"])
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Box + Points", "Bar + Error"])
        form.addRow("Group A Prefix", self.group_a_edit)
        form.addRow("Group B Prefix", self.group_b_edit)
        form.addRow("Analysis Scope", self.scope_combo)
        form.addRow("Metric", self.metric_combo)
        form.addRow("ROI Type", self.roi_kind_combo)
        form.addRow("Pair Mode", self.pair_mode_combo)
        form.addRow("Plot Type", self.plot_type_combo)
        layout.addLayout(form)

        control_row = QHBoxLayout()
        self.run_btn = QPushButton("Run Comparison")
        self.insert_chat_btn = QPushButton("Insert Summary to Chat")
        self.export_csv_btn = QPushButton("Export CSV")
        control_row.addWidget(self.run_btn)
        control_row.addWidget(self.insert_chat_btn)
        control_row.addWidget(self.export_csv_btn)
        control_row.addStretch(1)
        layout.addLayout(control_row)

        self.figure = Figure(figsize=(6.8, 3.4), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)
        layout.addWidget(self.canvas, 1)

        self.summary_label = QLabel("Descriptive stats and selected test will appear here.")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Group", "Sample", "ROI", "Metric", "Value"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)

        self.scope_combo.currentTextChanged.connect(self._sync_controls)
        self.metric_combo.currentTextChanged.connect(self._sync_controls)
        self.plot_type_combo.currentTextChanged.connect(self._update_plot)
        self.run_btn.clicked.connect(self.run_comparison)
        self.insert_chat_btn.clicked.connect(self.insert_summary_to_chat)
        self.export_csv_btn.clicked.connect(self.export_csv)

        self._sync_controls()

    def _sync_controls(self) -> None:
        roi_mode = self.scope_combo.currentText() == "ROI-based"
        self.roi_kind_combo.setEnabled(roi_mode)
        metric = self.metric_combo.currentText()
        if not roi_mode and metric == "area":
            self.metric_combo.setCurrentText("mean")

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _current_image_plane(self, layer: napari.layers.Image) -> np.ndarray:
        data = np.asarray(layer.data, dtype=float)
        if data.ndim < 2:
            raise ValueError(f"Layer [{layer.name}] must have at least 2 dimensions.")
        if data.ndim == 2:
            return data
        current_step = tuple(int(step) for step in self.viewer.dims.current_step[: data.ndim])
        leading_shape = data.shape[:-2]
        leading_indices = []
        for axis, axis_size in enumerate(leading_shape):
            step_index = current_step[axis] if axis < len(current_step) else 0
            leading_indices.append(int(np.clip(step_index, 0, axis_size - 1)))
        return np.asarray(data[tuple(leading_indices) + (slice(None), slice(None))], dtype=float)

    def _group_images(self, prefix: str) -> list[napari.layers.Image]:
        token = str(prefix or "").strip()
        if not token:
            return []
        return [
            layer
            for layer in self.viewer.layers
            if isinstance(layer, napari.layers.Image) and str(layer.name).startswith(token)
        ]

    def _strip_prefix(self, name: str, prefix: str) -> str:
        text = str(name or "")
        token = str(prefix or "")
        if token and text.startswith(token):
            text = text[len(token) :]
        return text.lstrip(" _-.") or str(name or "")

    def _resolve_matching_roi_layer(self, image_layer: napari.layers.Image, roi_kind: str):
        candidates = []
        for layer in self.viewer.layers:
            if layer is image_layer:
                continue
            if not str(getattr(layer, "name", "")).startswith(str(image_layer.name)):
                continue
            if roi_kind in {"auto", "shapes"} and isinstance(layer, napari.layers.Shapes):
                candidates.append(layer)
            if roi_kind in {"auto", "labels"} and isinstance(layer, napari.layers.Labels):
                candidates.append(layer)
        if roi_kind == "auto":
            for layer in candidates:
                if isinstance(layer, napari.layers.Shapes):
                    return layer
            for layer in candidates:
                if isinstance(layer, napari.layers.Labels):
                    return layer
        return candidates[0] if candidates else None

    def _roi_mask_from_layer(self, roi_layer, target_shape: tuple[int, ...]) -> np.ndarray:
        if isinstance(roi_layer, napari.layers.Labels):
            data = np.asarray(roi_layer.data) > 0
            if data.shape != target_shape:
                raise ValueError(
                    f"ROI labels shape {data.shape} does not match target image shape {target_shape}."
                )
            return data
        if isinstance(roi_layer, napari.layers.Shapes):
            masks = roi_layer.to_masks(mask_shape=target_shape)
            if not masks:
                return np.zeros(target_shape, dtype=bool)
            combined = np.zeros(target_shape, dtype=bool)
            for mask in masks:
                mask_array = np.asarray(mask, dtype=bool)
                if mask_array.shape == target_shape:
                    combined |= mask_array
            return combined
        raise ValueError("ROI layer must be Labels or Shapes.")

    def _metric_value(self, values: np.ndarray, metric: str) -> float:
        if metric == "mean":
            return float(np.mean(values))
        if metric == "median":
            return float(np.median(values))
        if metric == "sum":
            return float(np.sum(values))
        if metric == "std":
            return float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        if metric == "area":
            return float(values.size)
        raise ValueError(f"Unsupported metric [{metric}].")

    def _descriptive(self, values: list[float]) -> dict[str, float | int]:
        arr = np.asarray(values, dtype=float)
        return {
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    def _format_descriptive(self, name: str, stats_dict: dict[str, float | int]) -> str:
        return (
            f"{name}: n={stats_dict['n']} mean={stats_dict['mean']:.2f} std={stats_dict['std']:.2f} "
            f"median={stats_dict['median']:.2f} min={stats_dict['min']:.2f} max={stats_dict['max']:.2f}"
        )

    def _set_table_rows(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = [
                row.get("group", ""),
                row.get("sample", ""),
                row.get("roi", ""),
                row.get("metric", ""),
                f"{float(row.get('value', np.nan)):.2f}" if row.get("value") is not None else "",
            ]
            for column_index, value in enumerate(values):
                self.table.setItem(row_index, column_index, QTableWidgetItem(str(value)))
        self.table.resizeColumnsToContents()

    def _build_report(self, payload: dict[str, Any]) -> list[str]:
        lines = [
            f"{payload['scope']} comparison using metric [{payload['metric']}]",
            f"Group A prefix: [{payload['group_a_prefix']}]",
            f"Group B prefix: [{payload['group_b_prefix']}]",
            self._format_descriptive("Group A", payload["group_stats"]["A"]),
            self._format_descriptive("Group B", payload["group_stats"]["B"]),
        ]
        normality = payload.get("normality") or {}
        variance = payload.get("variance") or {}
        if normality:
            lines.append("Normality checks:")
            for name, info in normality.items():
                lines.append(f"- {name}: Shapiro-Wilk W={info['statistic']:.4g}, p={info['pvalue']:.4g}")
        if variance:
            lines.append(
                f"- Variance check: Levene statistic={variance['statistic']:.4g}, p={variance['pvalue']:.4g}"
            )
        lines.append(f"Selected test: {payload['selected_test']}")
        lines.append(
            f"Statistic={payload['result_stats']['statistic']:.4g}, p={payload['result_stats']['pvalue']:.4g}, "
            f"delta_mean={payload['result_stats']['delta_mean']:.4g}"
        )
        if "matched_pairs" in payload["result_stats"]:
            lines.append(f"Matched pairs={payload['result_stats']['matched_pairs']}")
        return lines

    def _summary_html(self) -> str:
        if not self._report_lines:
            return "Descriptive stats and selected test will appear here."
        head = "<br>".join(self._report_lines[:4])
        tail = "<br>".join(self._report_lines[4:])
        return f"{head}<br><br>{tail}" if tail else head

    def _update_plot(self) -> None:
        self.axes.clear()
        if not self._rows:
            self.axes.text(0.5, 0.5, "Run a comparison to show the plot.", ha="center", va="center", transform=self.axes.transAxes)
            self.axes.set_xticks([])
            self.axes.set_yticks([])
            self.canvas.draw_idle()
            return

        values_a = [float(row["value"]) for row in self._rows if row["group"] == "A"]
        values_b = [float(row["value"]) for row in self._rows if row["group"] == "B"]
        labels = ["Group A", "Group B"]
        plot_type = self.plot_type_combo.currentText()

        if plot_type == "Bar + Error":
            means = [float(np.mean(values_a)), float(np.mean(values_b))]
            errors = [
                float(np.std(values_a, ddof=1)) if len(values_a) > 1 else 0.0,
                float(np.std(values_b, ddof=1)) if len(values_b) > 1 else 0.0,
            ]
            colors = ["#4c72b0", "#dd8452"]
            self.axes.bar(labels, means, yerr=errors, capsize=6, color=colors, alpha=0.9)
            self.axes.set_title("Group Comparison: Bar + Error")
            self.axes.set_ylabel(self.metric_combo.currentText())
        else:
            positions = [1, 2]
            self.axes.boxplot([values_a, values_b], positions=positions, widths=0.45, patch_artist=True)
            rng = np.random.default_rng(0)
            jitter_a = positions[0] + rng.uniform(-0.07, 0.07, size=len(values_a))
            jitter_b = positions[1] + rng.uniform(-0.07, 0.07, size=len(values_b))
            self.axes.scatter(jitter_a, values_a, color="#4c72b0", s=35, zorder=3)
            self.axes.scatter(jitter_b, values_b, color="#dd8452", s=35, zorder=3)
            self.axes.set_xticks(positions)
            self.axes.set_xticklabels(labels)
            self.axes.set_title("Group Comparison: Box + Points")
            self.axes.set_ylabel(self.metric_combo.currentText())

        self.axes.grid(True, axis="y", alpha=0.25)
        self.canvas.draw_idle()

    def run_comparison(self) -> None:
        prefix_a = self.group_a_edit.text().strip()
        prefix_b = self.group_b_edit.text().strip()
        metric = self.metric_combo.currentText().strip()
        scope = self.scope_combo.currentText()
        roi_kind = self.roi_kind_combo.currentText().strip()
        pair_mode = self.pair_mode_combo.currentText().strip()
        alpha = 0.05

        if not prefix_a or not prefix_b:
            self._set_status("Both group prefixes are required.")
            return
        if scope != "ROI-based" and metric == "area":
            self._set_status("Area is only valid for ROI-based comparison.")
            return

        rows: list[dict[str, Any]] = []
        missing_roi: list[str] = []
        for group_name, prefix in (("A", prefix_a), ("B", prefix_b)):
            image_layers = self._group_images(prefix)
            for image_layer in image_layers:
                if getattr(image_layer, "rgb", False):
                    continue
                try:
                    image_plane = self._current_image_plane(image_layer)
                except Exception:
                    continue
                values = image_plane[np.isfinite(image_plane)]
                roi_name = ""
                if scope == "ROI-based":
                    roi_layer = self._resolve_matching_roi_layer(image_layer, roi_kind)
                    if roi_layer is None:
                        missing_roi.append(image_layer.name)
                        continue
                    roi_name = str(roi_layer.name)
                    try:
                        mask = self._roi_mask_from_layer(roi_layer, image_plane.shape)
                    except Exception:
                        continue
                    values = image_plane[mask]
                    values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                rows.append(
                    {
                        "group": group_name,
                        "sample": image_layer.name,
                        "roi": roi_name,
                        "metric": metric,
                        "value": self._metric_value(values, metric),
                        "pair_key": self._strip_prefix(image_layer.name, prefix),
                    }
                )

        if missing_roi:
            self._set_status(
                "Missing ROI layer for: " + ", ".join(missing_roi[:6]) + ("..." if len(missing_roi) > 6 else "")
            )
            self._set_table_rows([])
            self._report_lines = []
            self.summary_label.setText(self._summary_html())
            self._update_plot()
            return

        values_a = [float(row["value"]) for row in rows if row["group"] == "A"]
        values_b = [float(row["value"]) for row in rows if row["group"] == "B"]
        if len(values_a) < 2 or len(values_b) < 2:
            self._set_status("Each group needs at least 2 valid samples.")
            self._set_table_rows(rows)
            self._report_lines = []
            self.summary_label.setText(self._summary_html())
            self._update_plot()
            return

        arr_a = np.asarray(values_a, dtype=float)
        arr_b = np.asarray(values_b, dtype=float)
        normality: dict[str, dict[str, float]] = {}
        variance: dict[str, float] = {}
        selected_test = "Welch t-test"
        result_stats: dict[str, Any]

        if pair_mode == "paired_suffix":
            rows_a = {str(row["pair_key"]): float(row["value"]) for row in rows if row["group"] == "A"}
            rows_b = {str(row["pair_key"]): float(row["value"]) for row in rows if row["group"] == "B"}
            shared = sorted(set(rows_a) & set(rows_b))
            if len(shared) < 2:
                self._set_status("Paired mode needs at least 2 matched suffix pairs.")
                self._set_table_rows(rows)
                self._report_lines = []
                self.summary_label.setText(self._summary_html())
                self._update_plot()
                return
            paired_a = np.asarray([rows_a[key] for key in shared], dtype=float)
            paired_b = np.asarray([rows_b[key] for key in shared], dtype=float)
            diffs = paired_a - paired_b
            if diffs.size >= 3:
                shapiro = stats.shapiro(diffs)
                normality["paired_differences"] = {"statistic": float(shapiro.statistic), "pvalue": float(shapiro.pvalue)}
            if diffs.size >= 3 and normality["paired_differences"]["pvalue"] < alpha:
                selected_test = "Wilcoxon signed-rank"
                test = stats.wilcoxon(paired_a, paired_b, zero_method="wilcox", correction=False)
            else:
                selected_test = "Paired t-test"
                test = stats.ttest_rel(paired_a, paired_b, nan_policy="omit")
            result_stats = {
                "statistic": float(test.statistic),
                "pvalue": float(test.pvalue),
                "delta_mean": float(np.mean(paired_a) - np.mean(paired_b)),
                "matched_pairs": int(len(shared)),
            }
        else:
            if arr_a.size >= 3:
                shapiro_a = stats.shapiro(arr_a)
                normality["A"] = {"statistic": float(shapiro_a.statistic), "pvalue": float(shapiro_a.pvalue)}
            if arr_b.size >= 3:
                shapiro_b = stats.shapiro(arr_b)
                normality["B"] = {"statistic": float(shapiro_b.statistic), "pvalue": float(shapiro_b.pvalue)}
            if arr_a.size >= 2 and arr_b.size >= 2:
                lev = stats.levene(arr_a, arr_b, center="median")
                variance = {"statistic": float(lev.statistic), "pvalue": float(lev.pvalue)}
            normal_pass = all(info.get("pvalue", 1.0) >= alpha for info in normality.values()) if normality else False
            equal_var = variance.get("pvalue", 0.0) >= alpha if variance else False
            if normal_pass:
                if equal_var:
                    selected_test = "Student t-test"
                    test = stats.ttest_ind(arr_a, arr_b, equal_var=True, nan_policy="omit")
                else:
                    selected_test = "Welch t-test"
                    test = stats.ttest_ind(arr_a, arr_b, equal_var=False, nan_policy="omit")
            else:
                selected_test = "Mann-Whitney U"
                test = stats.mannwhitneyu(arr_a, arr_b, alternative="two-sided")
            result_stats = {
                "statistic": float(test.statistic),
                "pvalue": float(test.pvalue),
                "delta_mean": float(np.mean(arr_a) - np.mean(arr_b)),
            }

        payload = {
            "scope": "ROI-based" if scope == "ROI-based" else "Whole-image",
            "metric": metric,
            "group_a_prefix": prefix_a,
            "group_b_prefix": prefix_b,
            "group_stats": {"A": self._descriptive(values_a), "B": self._descriptive(values_b)},
            "normality": normality,
            "variance": variance,
            "selected_test": selected_test,
            "result_stats": result_stats,
        }
        self._set_table_rows(rows)
        self._report_lines = self._build_report(payload)
        self.summary_label.setText(self._summary_html())
        self._update_plot()
        self._set_status(
            f"Compared {len(values_a)} sample(s) in group A and {len(values_b)} sample(s) in group B using {selected_test}."
        )

    def _summary_text(self) -> str:
        return "\n".join(self._report_lines).strip()

    def insert_summary_to_chat(self) -> None:
        content = self._summary_text()
        if not content:
            self._set_status("Run a comparison first.")
            return
        prompt = _find_prompt_widget()
        if prompt is None:
            self._set_status("Chat prompt was not found. Use Copy from the widget or export CSV.")
            return
        current = prompt.toPlainText().rstrip()
        prompt.setPlainText(f"{current}\n{content}" if current else content)
        prompt.setFocus()
        self._set_status("Inserted group comparison summary into chat.")

    def export_csv(self) -> None:
        if not self._rows:
            self._set_status("Run a comparison first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Group Comparison Dataset",
            "group_comparison_stats.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Group", "Sample", "ROI", "Metric", "Value"])
            for row in self._rows:
                writer.writerow([row["group"], row["sample"], row["roi"], row["metric"], f"{float(row['value']):.6g}"])
            if self._report_lines:
                writer.writerow([])
                writer.writerow(["Summary"])
                for line in self._report_lines:
                    writer.writerow([line])
        self._set_status(f"Exported {len(self._rows)} sample row(s) to {path}.")


def open_group_comparison_widget(viewer: napari.Viewer):
    existing_widget = getattr(viewer.window, "dock_widgets", {}).get(WIDGET_NAME)
    if existing_widget is not None:
        try:
            existing_dock = _dock_wrapper_from_widget(existing_widget)
            if existing_dock is None:
                raise RuntimeError("Dock wrapper not found.")
            existing_dock.setFloating(True)
            existing_dock.show()
            existing_dock.raise_()
            return existing_widget
        except Exception:
            pass

    widget = GroupComparisonWidget(viewer)
    dock_widget = viewer.window.add_dock_widget(widget, name=WIDGET_NAME, area="right")
    try:
        dock_widget.setFloating(True)
        dock_widget.show()
        dock_widget.raise_()
    except Exception:
        pass
    return widget
