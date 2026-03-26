from __future__ import annotations

import os

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class _QtBotStub:
    def wait_exposed(self, *_args, **_kwargs):
        return None

    def wait(self, *_args, **_kwargs):
        return None


@pytest.fixture
def qtbot():
    return _QtBotStub()
