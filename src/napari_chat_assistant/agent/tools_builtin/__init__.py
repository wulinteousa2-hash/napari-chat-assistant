from __future__ import annotations

from .enhancement import ApplyClaheTool
from .workbench import workbench_scaffold_tools


def builtin_tools():
    tools = [
        ApplyClaheTool(),
    ]
    tools.extend(workbench_scaffold_tools())
    return tools
