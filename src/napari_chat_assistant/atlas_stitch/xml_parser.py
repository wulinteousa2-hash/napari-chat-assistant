from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import Any
import xml.etree.ElementTree as ET

from .models import AtlasMetadata, AtlasProject, TileRecord, TileTransform


_PATH_KEYS = (
    "path",
    "filepath",
    "file_path",
    "filename",
    "file",
    "url",
    "tilepath",
    "tile_path",
    "image",
    "imagefile",
    "image_path",
    "imagepath",
    "src",
    "source",
)


def parse_atlas_xml(xml_path: str, tile_root_override: str | None = None) -> AtlasProject:
    xml_file = Path(xml_path).expanduser().resolve()
    tree = ET.parse(xml_file)
    root = tree.getroot()
    override = str(tile_root_override or "").strip()

    metadata_values = _collect_metadata(root)
    pixel_size_value, pixel_size_unit = _pixel_size_fields(root)
    voxel_size_x = _pick_first_float(metadata_values, ("voxel_size_x", "pixel_size_x", "pixelsizex", "spacingx"))
    voxel_size_y = _pick_first_float(metadata_values, ("voxel_size_y", "pixel_size_y", "pixelsizey", "spacingy"))
    voxel_size_z = _pick_first_float(metadata_values, ("voxel_size_z", "pixel_size_z", "pixelsizez", "spacingz"))
    if pixel_size_value is not None:
        if voxel_size_x is None:
            voxel_size_x = pixel_size_value
        if voxel_size_y is None:
            voxel_size_y = pixel_size_value
    if pixel_size_unit and "pixel_size_unit" not in metadata_values:
        metadata_values["pixel_size_unit"] = pixel_size_unit
    atlas_name = _pick_first_text(metadata_values, ("atlas_name", "name", "dataset_name", "title")) or xml_file.stem
    metadata = AtlasMetadata(
        atlas_name=atlas_name,
        xml_path=str(xml_file),
        source_directory=str(xml_file.parent),
        tile_root_override=override,
        source_software=_pick_first_text(metadata_values, ("software", "application", "generator", "vendor")),
        image_width=_pick_first_int(metadata_values, ("image_width", "width", "sizex")),
        image_height=_pick_first_int(metadata_values, ("image_height", "height", "sizey")),
        image_depth=_pick_first_int(metadata_values, ("image_depth", "depth", "sizez")),
        channel_count=_pick_first_int(metadata_values, ("channel_count", "channels", "sizec")),
        voxel_size_x=voxel_size_x,
        voxel_size_y=voxel_size_y,
        voxel_size_z=voxel_size_z,
        extra_metadata=metadata_values,
    )

    tiles: list[TileRecord] = []
    seen_signatures: set[tuple[int | None, int | None, str, float | None, float | None]] = set()
    warnings: list[str] = []
    duplicate_count = 0
    ignored_element_count = 0
    for index, element in enumerate(root.iter()):
        raw_values = _element_value_map(element)
        row = _pick_first_int(raw_values, ("row", "row_index", "r"))
        col = _pick_first_int(raw_values, ("column", "col", "column_index"))
        source_path = _pick_first_text(raw_values, _PATH_KEYS)
        if row is None or col is None or not source_path:
            ignored_element_count += 1
            continue
        tile = _parse_tile_element(
            element,
            index=index,
            xml_dir=xml_file.parent,
            tile_root_override=override,
            raw_values=raw_values,
        )
        if tile is None:
            continue
        signature = (
            tile.row,
            tile.col,
            tile.file_name,
            tile.start_x,
            tile.start_y,
        )
        if signature in seen_signatures:
            duplicate_count += 1
            continue
        seen_signatures.add(signature)
        tiles.append(tile)

    metadata.tile_count = len(tiles)
    metadata.extra_metadata["ignored_non_tile_elements"] = ignored_element_count
    metadata.extra_metadata["duplicate_tile_elements"] = duplicate_count
    if duplicate_count:
        warnings.append(
            f"Ignored {duplicate_count} duplicate tile XML element(s) while collecting atlas tiles."
        )
    if ignored_element_count:
        warnings.append(
            f"Ignored {ignored_element_count} non-tile XML element(s) during parsing."
        )
    missing_tiles = [tile.file_name or tile.tile_id for tile in tiles if not tile.exists]
    return AtlasProject(metadata=metadata, tiles=tiles, missing_tiles=missing_tiles, warnings=warnings)


def _collect_metadata(root: ET.Element) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for element in root.iter():
        tag = _clean_key(element.tag)
        if tag and element.text and element.text.strip() and tag not in values:
            values[tag] = element.text.strip()
        for key, value in element.attrib.items():
            clean_key = _clean_key(key)
            if clean_key and value.strip() and clean_key not in values:
                values[clean_key] = value.strip()
    return values


def _pixel_size_fields(root: ET.Element) -> tuple[float | None, str]:
    for element in root.iter():
        if _clean_key(element.tag) != "pixelsize":
            continue
        nested = _element_value_map(element)
        value = _pick_first_float(nested, ("value", "pixel_size", "pixelsize"))
        unit = _pick_first_text(nested, ("unit", "pixel_size_unit", "pixelsizeunit"))
        if value is not None or unit:
            return value, unit
    return None, ""


def _parse_tile_element(
    element: ET.Element,
    *,
    index: int,
    xml_dir: Path,
    tile_root_override: str,
    raw_values: dict[str, str] | None = None,
    ) -> TileRecord | None:
    raw_values = raw_values or _element_value_map(element)
    source_path = _pick_first_text(raw_values, _PATH_KEYS)
    if not source_path:
        return None

    resolved_path = _resolve_tile_path(source_path, xml_dir=xml_dir, tile_root_override=tile_root_override)
    tile_id = (
        _pick_first_text(raw_values, ("tile_id", "tileid", "id", "name", "label"))
        or f"tile_{index:04d}"
    )
    file_name = Path(source_path).name
    raw_values.setdefault("file_name", file_name)
    raw_values.setdefault("source_path", source_path)
    row = _pick_first_int(raw_values, ("row", "row_index", "r"))
    col = _pick_first_int(raw_values, ("column", "col", "column_index"))
    start_x = _pick_first_float(raw_values, ("start_x", "startx", "left", "originx"))
    start_y = _pick_first_float(raw_values, ("start_y", "starty", "top", "originy"))
    position_x = _pick_first_float(raw_values, ("position_x", "posx", "tile_position_x", "stagex"))
    position_y = _pick_first_float(raw_values, ("position_y", "posy", "tile_position_y", "stagey"))
    position_z = _pick_first_float(raw_values, ("position_z", "posz", "tile_position_z", "stagez"))
    transform = TileTransform(
        nominal_x=start_x if start_x is not None else position_x or 0.0,
        nominal_y=start_y if start_y is not None else position_y or 0.0,
        nominal_z=position_z or 0.0,
        scale_x=_pick_first_float(raw_values, ("scale_x", "scalex")) or 1.0,
        scale_y=_pick_first_float(raw_values, ("scale_y", "scaley")) or 1.0,
        scale_z=_pick_first_float(raw_values, ("scale_z", "scalez")) or 1.0,
        rotation_degrees=_pick_first_float(raw_values, ("rotation", "angle", "rotation_degrees")) or 0.0,
    )
    return TileRecord(
        tile_id=tile_id,
        file_name=file_name,
        source_path=source_path,
        resolved_path=str(resolved_path) if resolved_path is not None else "",
        row=row,
        col=col,
        start_x=start_x,
        start_y=start_y,
        position_x=position_x,
        position_y=position_y,
        width=_pick_first_int(raw_values, ("width", "sizex")),
        height=_pick_first_int(raw_values, ("height", "sizey")),
        exists=resolved_path is not None and resolved_path.exists(),
        transform=transform,
        metadata=raw_values,
    )


def _element_value_map(element: ET.Element) -> dict[str, str]:
    values: dict[str, str] = {}
    for key, value in element.attrib.items():
        if value.strip():
            values[_clean_key(key)] = value.strip()
    for child in element:
        if child.text and child.text.strip():
            values.setdefault(_clean_key(child.tag), child.text.strip())
        for key, value in child.attrib.items():
            if value.strip():
                values.setdefault(_clean_key(key), value.strip())
    if element.text and element.text.strip():
        values.setdefault(_clean_key(element.tag), element.text.strip())
    return values


def _resolve_tile_path(source_path: str, *, xml_dir: Path, tile_root_override: str) -> Path | None:
    if not source_path.strip():
        return None
    text = source_path.strip().strip('"').strip("'")
    posix_text = text.replace("\\", "/")

    candidates: list[Path] = []
    if tile_root_override.strip():
        override_root = Path(tile_root_override).expanduser()
        override_candidates = _override_relative_parts(text)
        for part in override_candidates:
            candidates.append(override_root / part)
        candidates.append(override_root / Path(posix_text).name)

    source_path_obj = Path(posix_text).expanduser()
    candidates.append(source_path_obj)
    candidates.append(xml_dir / source_path_obj)
    candidates.append(xml_dir / Path(posix_text).name)

    for candidate in candidates:
        try:
            normalized = candidate.resolve(strict=False)
        except Exception:
            normalized = candidate
        if normalized.exists():
            return normalized

    return candidates[0].resolve(strict=False) if candidates else None


def _override_relative_parts(path_text: str) -> list[Path]:
    win_path = PureWindowsPath(path_text)
    parts = [part for part in win_path.parts if part not in (win_path.anchor, "\\", "/")]
    filtered_parts = [part for part in parts if ":" not in part]
    if not filtered_parts:
        posix_name = Path(path_text.replace("\\", "/")).name
        return [Path(posix_name)] if posix_name else []
    candidates = [Path(*filtered_parts)]
    if len(filtered_parts) > 1:
        candidates.append(Path(*filtered_parts[-2:]))
    candidates.append(Path(filtered_parts[-1]))
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _clean_key(value: str) -> str:
    text = value.rsplit("}", 1)[-1]
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def _pick_first_text(values: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = values.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _pick_first_int(values: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = values.get(key)
        if value is None or str(value).strip() == "":
            continue
        try:
            return int(float(str(value).strip()))
        except (TypeError, ValueError):
            continue
    return None


def _pick_first_float(values: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = values.get(key)
        if value is None or str(value).strip() == "":
            continue
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            continue
    return None
