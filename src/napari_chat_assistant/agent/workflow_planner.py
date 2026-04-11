from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


StepKind = Literal["analysis", "decision", "tool", "stop_check"]


@dataclass(frozen=True)
class WorkflowConstraint:
    id: str
    description: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class WorkflowStep:
    id: str
    kind: StepKind
    title: str
    rationale: str
    tool: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    condition: str = ""
    on_error: str = "stop"
    output_alias: str = ""
    success_checks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkflowPlan:
    workflow_type: str
    intent: str
    target_layer: str
    target_layer_type: str
    constraints: list[WorkflowConstraint]
    assumptions: list[str]
    steps: list[WorkflowStep]
    stop_conditions: list[str]
    planner_notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_type": self.workflow_type,
            "intent": self.intent,
            "target_layer": self.target_layer,
            "target_layer_type": self.target_layer_type,
            "constraints": [item.to_dict() for item in self.constraints],
            "assumptions": list(self.assumptions),
            "steps": [step.to_dict() for step in self.steps],
            "stop_conditions": list(self.stop_conditions),
            "planner_notes": list(self.planner_notes),
        }


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def looks_like_conservative_binary_segmentation_request(text: str) -> bool:
    source = _normalize_text(text)
    if not source:
        return False
    segmentation_terms = (
        "binary mask",
        "foreground objects",
        "threshold",
        "mask quality",
        "main foreground",
        "preview threshold",
        "selected image",
        "measure quality",
        "measure mask quality",
    )
    conservative_terms = (
        "conservative workflow",
        "conservative binary mask",
        "build a conservative binary mask",
        "clean the mask minimally",
        "minimum necessary morphology",
        "preserve faint real structures",
        "light denoising only if needed",
        "do not destroy faint",
        "avoid over-merging",
        "cleanest possible",
    )
    return sum(1 for term in segmentation_terms if term in source) >= 3 and _has_any(source, conservative_terms)


def plan_conservative_binary_segmentation(
    text: str,
    *,
    selected_layer_profile: dict[str, Any] | None = None,
) -> WorkflowPlan | None:
    if not looks_like_conservative_binary_segmentation_request(text):
        return None

    profile = selected_layer_profile if isinstance(selected_layer_profile, dict) else {}
    target_layer = str(profile.get("layer_name") or "selected_image").strip() or "selected_image"
    target_layer_type = str(profile.get("layer_type") or "image").strip().lower() or "image"
    semantic_type = str(profile.get("semantic_type") or "").strip().lower()

    assumptions = [
        f"Target image layer: [{target_layer}].",
        "Threshold direction should stay automatic until inspection indicates a clear bright-foreground or dim-foreground case.",
        "Mask cleanup should create new layers rather than overwrite the first threshold result.",
    ]
    if semantic_type:
        assumptions.append(f"Selected layer semantic type was profiled as [{semantic_type}].")

    constraints = [
        WorkflowConstraint("prefer_builtin_tools", "Prefer registered built-in tools over generated code."),
        WorkflowConstraint("preview_before_apply", "Preview threshold before committing an applied mask."),
        WorkflowConstraint("preserve_faint_signal", "Do not destroy faint real structures just to reduce noise."),
        WorkflowConstraint("minimum_necessary_morphology", "Use the minimum necessary morphology and avoid over-merging nearby objects."),
        WorkflowConstraint("explain_each_change", "Explain what changed and why after each major step."),
    ]

    steps = [
        WorkflowStep(
            id="inspect_selected_image",
            kind="analysis",
            title="Inspect the selected image first",
            rationale="Estimate likely signal structure, background, and noise before choosing denoising or threshold direction.",
            success_checks=[
                "Describe likely foreground signal.",
                "Describe background pattern.",
                "Describe visible noise or speckle level.",
            ],
        ),
        WorkflowStep(
            id="decide_light_denoise",
            kind="decision",
            title="Decide whether light denoising is needed",
            rationale="The prompt explicitly requests denoising only when the image quality justifies it.",
            condition="If noise or speckle would destabilize thresholding, denoise lightly; otherwise keep the raw image.",
            success_checks=[
                "State whether denoising is needed.",
                "If yes, justify a conservative strength.",
            ],
        ),
        WorkflowStep(
            id="gaussian_denoise_if_needed",
            kind="tool",
            title="Apply light Gaussian denoising if needed",
            rationale="Use a conservative smoothing step only when inspection suggests thresholding would otherwise be unstable.",
            tool="gaussian_denoise",
            arguments={"layer_name": target_layer, "sigma": 1.0, "preserve_range": True},
            condition="Run only if the previous decision concludes that light denoising is needed.",
            on_error="skip",
            output_alias="working_image",
            success_checks=["Create a lightly denoised working image without erasing faint foreground structure."],
        ),
        WorkflowStep(
            id="choose_threshold_direction",
            kind="decision",
            title="Choose threshold direction conservatively",
            rationale="Threshold direction must match whether the foreground is brighter or dimmer than the background.",
            condition="If foreground is brighter than the background use bright-foreground thresholding; if dimmer use dim-foreground thresholding; otherwise start with automatic thresholding and explain the uncertainty.",
            success_checks=["State whether the foreground appears brighter or dimmer than the background and give the evidence."],
        ),
        WorkflowStep(
            id="preview_threshold",
            kind="tool",
            title="Preview threshold instead of applying immediately",
            rationale="The prompt requires preview-first behavior to avoid committing a poor mask too early.",
            tool="preview_threshold",
            arguments={"layer_name": "$working_image_or_selected", "polarity": "auto"},
            on_error="stop",
            output_alias="threshold_preview",
            success_checks=["Produce a preview mask and summarize object count and foreground coverage."],
        ),
        WorkflowStep(
            id="review_threshold_preview",
            kind="analysis",
            title="Review threshold preview quality",
            rationale="The preview should be judged before committing any mask layer.",
            success_checks=[
                "State whether the preview is too loose, too strict, or acceptable.",
                "Call out likely missing faint structures or background leakage.",
            ],
        ),
        WorkflowStep(
            id="apply_threshold",
            kind="tool",
            title="Apply the chosen threshold",
            rationale="After preview review, create a real labels mask layer for conservative cleanup.",
            tool="apply_threshold",
            arguments={"layer_name": "$working_image_or_selected", "polarity": "auto"},
            on_error="stop",
            output_alias="base_mask",
            success_checks=["Create a binary labels layer for the selected image."],
        ),
        WorkflowStep(
            id="measure_base_mask",
            kind="tool",
            title="Measure the first applied mask",
            rationale="The prompt asks to summarize mask quality after each major step.",
            tool="measure_mask",
            arguments={"layer_name": "$base_mask"},
            on_error="skip",
            success_checks=["Report object count, foreground coverage, and major obvious defects."],
        ),
        WorkflowStep(
            id="remove_tiny_specks_if_needed",
            kind="tool",
            title="Remove tiny specks conservatively",
            rationale="This handles isolated noise without changing larger real structures.",
            tool="remove_small_objects",
            arguments={"layer_name": "$base_mask", "min_size": 64},
            condition="Run only if the mask shows speckle-like tiny connected components.",
            on_error="skip",
            output_alias="mask_without_specks",
            success_checks=["Create a cleaned mask with small isolated noise removed."],
        ),
        WorkflowStep(
            id="fill_small_holes_if_needed",
            kind="tool",
            title="Fill small internal holes if needed",
            rationale="Small holes can be cleaned conservatively without merging nearby objects.",
            tool="fill_mask_holes",
            arguments={"layer_name": "$latest_mask"},
            condition="Run only if biologically implausible holes remain inside otherwise solid foreground objects.",
            on_error="skip",
            output_alias="mask_filled",
            success_checks=["Create a hole-filled mask while preserving object separation."],
        ),
        WorkflowStep(
            id="smooth_jagged_edges_if_needed",
            kind="tool",
            title="Smooth jagged edges conservatively",
            rationale="A light binary median pass can reduce jagged boundaries with lower merge risk than aggressive closing.",
            tool="run_mask_op",
            arguments={"layer_name": "$latest_mask", "op": "median", "radius": 1},
            condition="Run only if object edges are visibly jagged and nearby objects are not at risk of merging.",
            on_error="skip",
            output_alias="mask_smoothed",
            success_checks=["Create a slightly smoother mask without obvious over-merging."],
        ),
        WorkflowStep(
            id="measure_cleaned_mask",
            kind="tool",
            title="Measure the cleaned mask again",
            rationale="Mask quality should be summarized after each major cleanup stage.",
            tool="measure_mask",
            arguments={"layer_name": "$latest_mask"},
            on_error="skip",
            success_checks=["Report whether noise decreased without losing plausible faint structures."],
        ),
        WorkflowStep(
            id="refine_if_too_loose_or_strict",
            kind="decision",
            title="Refine iteratively only if needed",
            rationale="The workflow should loop only when the current mask is clearly too loose or too strict.",
            condition="If the mask is too loose, tighten threshold or cleanup slightly; if too strict, reduce cleanup aggression or relax the threshold. Avoid repeating large changes.",
            success_checks=[
                "Explain whether another refinement cycle is needed.",
                "If yes, specify the single next adjustment and why.",
            ],
        ),
        WorkflowStep(
            id="stop_when_conservative_quality_is_met",
            kind="stop_check",
            title="Stop when the mask is clean and plausible",
            rationale="The user asked for a biologically plausible, low-noise mask while preserving faint real signal.",
            success_checks=[
                "Mask is low noise.",
                "Mask is biologically plausible.",
                "Faint real structures are not obviously erased.",
                "Cleanup has not over-merged nearby objects.",
            ],
        ),
    ]

    stop_conditions = [
        "The mask is clean enough for downstream use.",
        "Noise and speckle are low without obvious destruction of faint real structures.",
        "Foreground extent is biologically plausible.",
        "No further conservative refinement is clearly justified.",
    ]
    planner_notes = [
        "This is a planner-only phase. It structures the workflow before adding a mixed analysis-and-tool executor.",
        "Tool arguments include placeholders such as `$latest_mask` and `$working_image_or_selected` because later execution should resolve intermediate outputs dynamically.",
    ]

    return WorkflowPlan(
        workflow_type="conservative_binary_segmentation",
        intent="Inspect the selected image and build a conservative binary foreground mask with iterative quality checks.",
        target_layer=target_layer,
        target_layer_type=target_layer_type,
        constraints=constraints,
        assumptions=assumptions,
        steps=steps,
        stop_conditions=stop_conditions,
        planner_notes=planner_notes,
    )


def workflow_plan_to_markdown(plan: WorkflowPlan | dict[str, Any]) -> str:
    payload = plan.to_dict() if isinstance(plan, WorkflowPlan) else dict(plan or {})
    workflow_type = str(payload.get("workflow_type", "")).strip() or "workflow"
    intent = str(payload.get("intent", "")).strip()
    target_layer = str(payload.get("target_layer", "")).strip()
    lines = [f"Planned workflow: `{workflow_type}`"]
    if intent:
        lines.append(intent)
    if target_layer:
        lines.append(f"Target layer: [{target_layer}]")

    constraints = payload.get("constraints", [])
    if isinstance(constraints, list) and constraints:
        lines.append("")
        lines.append("Constraints:")
        for item in constraints:
            if isinstance(item, dict):
                lines.append(f"- {str(item.get('description', '')).strip()}")

    steps = payload.get("steps", [])
    if isinstance(steps, list) and steps:
        lines.append("")
        lines.append("Planned steps:")
        for index, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                continue
            title = str(step.get("title", "")).strip() or f"Step {index}"
            kind = str(step.get("kind", "")).strip()
            tool = str(step.get("tool", "")).strip()
            rationale = str(step.get("rationale", "")).strip()
            suffix = f" using `{tool}`" if kind == "tool" and tool else ""
            lines.append(f"{index}. {title}{suffix}")
            if rationale:
                lines.append(f"   Why: {rationale}")
            condition = str(step.get("condition", "")).strip()
            if condition:
                lines.append(f"   Condition: {condition}")

    stop_conditions = payload.get("stop_conditions", [])
    if isinstance(stop_conditions, list) and stop_conditions:
        lines.append("")
        lines.append("Stop when:")
        for item in stop_conditions:
            text = str(item).strip()
            if text:
                lines.append(f"- {text}")

    notes = payload.get("planner_notes", [])
    if isinstance(notes, list) and notes:
        lines.append("")
        lines.append("Planner notes:")
        for item in notes:
            text = str(item).strip()
            if text:
                lines.append(f"- {text}")

    return "\n".join(lines).strip()
