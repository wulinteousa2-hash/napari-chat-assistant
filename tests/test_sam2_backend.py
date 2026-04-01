from __future__ import annotations

from napari_chat_assistant.agent.sam2_backend import _load_wrapper_module, SAM2BackendConfig, discover_sam2_setup


def test_discover_sam2_setup_finds_checkpoint_and_config_without_external_wrapper(tmp_path):
    project = tmp_path / "sam2"
    checkpoints = project / "checkpoints"
    configs = project / "configs" / "sam2.1"
    checkpoints.mkdir(parents=True)
    configs.mkdir(parents=True)
    (checkpoints / "sam2.1_hiera_large.pt").write_text("weights", encoding="utf-8")
    (configs / "sam2.1_hiera_l.yaml").write_text("model: test\n", encoding="utf-8")

    detected, message = discover_sam2_setup(
        {
            "sam2_project_path": str(project),
            "sam2_checkpoint_path": "",
            "sam2_config_path": "",
            "sam2_device": "cuda",
        }
    )

    assert detected["sam2_project_path"] == str(project)
    assert detected["sam2_checkpoint_path"] == "checkpoints/sam2.1_hiera_large.pt"
    assert detected["sam2_config_path"] == "configs/sam2.1/sam2.1_hiera_l.yaml"
    assert detected["sam2_device"] == "cuda"
    assert "Adapter: bundled [napari_chat_assistant.integrations.sam2_adapter]" in message
    assert "Checkpoint: [checkpoints/sam2.1_hiera_large.pt]" in message
    assert "Config: [configs/sam2.1/sam2.1_hiera_l.yaml]" in message


def test_load_wrapper_module_falls_back_to_bundled_adapter(tmp_path):
    project = tmp_path / "sam2"
    project.mkdir()

    module = _load_wrapper_module(
        SAM2BackendConfig(
            project_path=str(project),
            checkpoint_path="checkpoints/sam2.1_hiera_large.pt",
            config_path="configs/sam2.1/sam2.1_hiera_l.yaml",
            device="cpu",
        )
    )

    assert module.__name__ == "napari_chat_assistant.integrations.sam2_adapter"
    assert hasattr(module, "segment_image_from_box")
    assert hasattr(module, "segment_image_from_points")
