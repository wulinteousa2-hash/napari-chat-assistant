# Assistant Reference

## Purpose

`napari-chat-assistant` is a local Ollama-powered assistant for napari. It is designed to help users inspect the current session, automate common image-analysis tasks, and orchestrate tool-backed workflows from chat.

This assistant is not intended to be a free-form general chatbot. It is meant to operate inside napari with explicit, limited, inspectable actions.

## Current Model Backend

- Local provider: Ollama
- Default local URL: `http://127.0.0.1:11434`
- Model is selected from local Ollama tags or typed manually

## Core Design

The assistant works through a constrained pattern:

1. Read a compact summary of the current napari session.
2. Send that context and the user request to a local model.
3. Interpret the model response as either:
   - a normal reply
   - or a tool request
4. Execute tool-backed actions locally in napari.
5. Report the result back in chat.

## Current Tool Set

### Session Inspection

- `list_layers`
  - returns a text summary of the current napari layers

### Thresholding

- `preview_threshold`
  - target: one `Image` layer
  - creates or updates a preview `Labels` layer

- `apply_threshold`
  - target: one `Image` layer
  - creates a new output `Labels` layer

- `preview_threshold_batch`
  - target: all open `Image` layers
  - creates one preview mask per image layer

- `apply_threshold_batch`
  - target: all open `Image` layers
  - creates one output labels layer per image layer

### Mask Measurement

- `measure_mask`
  - target: one `Labels` layer
  - reports foreground count, object count, largest object, and area/volume when possible

- `measure_mask_batch`
  - target: all open `Labels` layers
  - reports measurements for all labels layers

### Mask Cleanup

- `run_mask_op`
  - target: one `Labels` layer
  - supported operations:
    - `dilate`
    - `erode`
    - `open`
    - `close`
    - `fill_holes`
    - `remove_small`
    - `keep_largest`

- `run_mask_op_batch`
  - target: all open `Labels` layers
  - applies one cleanup operation across all labels layers

## Layer Expectations

- Threshold tools operate on `Image` layers.
- Mask cleanup and measurement tools operate on `Labels` layers.
- If no layer name is specified for a single-layer tool:
  - the selected compatible layer is preferred
  - otherwise the first compatible layer is used

## Batch Behavior

Batch tools are intended for requests such as:
- `all images`
- `every image`
- `multiple open images`
- `batch`

Current batch rules:
- threshold batch iterates over all `Image` layers
- mask operation batch iterates over all `Labels` layers
- measurement batch iterates over all `Labels` layers

## Naming Conventions

### Preview Layers

- single-image preview:
  - `__assistant_threshold_preview__`

- batch preview:
  - `__assistant_threshold_preview__::<image_layer_name>`

### Threshold Output Layers

- default output:
  - `<image_layer_name>_labels`

- if that name already exists:
  - suffixes are appended, such as:
    - `<image_layer_name>_labels_01`
    - `<image_layer_name>_labels_02`

### Snapshot Layers

- before a mask edit, a snapshot is created as:
  - `<labels_layer_name>_assistant_snapshot_01`
  - `<labels_layer_name>_assistant_snapshot_02`

## Safety Behavior

- mask operations create a snapshot before modifying a labels layer
- tool execution is explicit and logged in the UI
- assistant-triggered heavy operations run in workers

## 2D / 3D Notes

- the assistant can inspect and operate on 2D or 3D array-backed layers as they exist in napari
- measurement reports:
  - area for 2D masks
  - volume for 3D masks
- current behavior is still basic and should be treated cautiously for large EM volumes

## Current Limitations

- early-stage architecture and workflow coverage
- no multimodal image prompt path yet
- no code-generation execution workflow yet
- no advanced planning loop
- limited toolset compared with full ImageJ/Fiji workflows
- batch behavior is layer-wide, not yet filtered by naming rules or selection sets

## Intended Direction

The assistant is moving toward:
- stronger automation
- code generation for napari scripting
- safer batch workflows
- better 2D/3D EM handling
- clearer tool orchestration rather than generic conversation
