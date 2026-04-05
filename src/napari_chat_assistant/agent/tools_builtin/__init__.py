from __future__ import annotations

from .enhancement import ApplyClaheTool
from .legacy_surface import legacy_surface_tools
from .workbench import workbench_scaffold_tools


def builtin_tools():
    tools = [
        ApplyClaheTool(),
    ]
    tools.extend(legacy_surface_tools())
    tools.extend(workbench_scaffold_tools())
    return tools
