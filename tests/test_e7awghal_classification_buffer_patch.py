# ruff: noqa: SLF001

"""Tests for e7awghal classification-buffer patch."""

from __future__ import annotations

from types import SimpleNamespace

from qubex.patches.quel_ic_config import e7awghal_classification_buffer_patch as patch


class _FakeMemoryManager:
    def __init__(self) -> None:
        self.calls: list[tuple[object, int, dict[str, object]]] = []

    def allocate(self, bufsize, *, minimum_align: int, **kwargs):
        self.calls.append((bufsize, minimum_align, kwargs))
        return SimpleNamespace(
            bufsize=bufsize, minimum_align=minimum_align, kwargs=kwargs
        )


class _FakeCaptureParam:
    def __init__(self, *, classification_enable: bool, datasize_in_sample: int) -> None:
        self.classification_enable = classification_enable
        self._datasize_in_sample = datasize_in_sample

    def get_datasize_in_sample(self) -> int:
        return self._datasize_in_sample


class _FakeCapUnitSimplified:
    def __init__(self) -> None:
        self._mm = _FakeMemoryManager()
        self._current_param = None
        self.original_calls = 0

    def _allocate_read_buffer(self, **kwargs):
        if self._current_param is None:
            raise AssertionError("_allocate_read_buffer() requires self._current_param")
        self.original_calls += 1
        if self._current_param.classification_enable:
            bufsize = self._current_param.get_datasize_in_sample() / 4
        else:
            bufsize = self._current_param.get_datasize_in_sample() * 8
        return self._mm.allocate(bufsize, minimum_align=64, **kwargs)


def test_apply_patch_does_not_fail_when_e7awghal_missing(monkeypatch) -> None:
    """Given missing e7awghal, when patch applies, then no exception is raised."""
    monkeypatch.setattr(patch, "_is_quelware_0_10_or_later", lambda: True)
    import_calls: list[str] = []

    def _raise_import_error(name: str):
        import_calls.append(name)
        raise ImportError(name)

    monkeypatch.setattr(patch.importlib, "import_module", _raise_import_error)

    patch.apply_e7awghal_classification_buffer_patch()

    assert import_calls == ["e7awghal.capunit"]


def test_apply_patch_skips_for_quelware_0_8(monkeypatch) -> None:
    """Given quelware 0.8.x, when patch applies, then e7awghal is not imported."""
    monkeypatch.setattr(patch, "_is_quelware_0_10_or_later", lambda: False)
    import_calls: list[str] = []

    def _import_module(name: str):
        import_calls.append(name)
        raise AssertionError("import must not be called for quelware 0.8.x")

    monkeypatch.setattr(patch.importlib, "import_module", _import_module)

    patch.apply_e7awghal_classification_buffer_patch()

    assert import_calls == []


def test_apply_patch_integerizes_classification_buffer_size(monkeypatch) -> None:
    """Given DSP classification, when buffer allocates, then byte count is an integer."""
    capunit_module = SimpleNamespace(
        CapUnitSimplified=_FakeCapUnitSimplified,
        _CAP_MINIMUM_ALIGN=64,
    )
    monkeypatch.setattr(patch, "_is_quelware_0_10_or_later", lambda: True)
    monkeypatch.setattr(
        patch.importlib,
        "import_module",
        lambda name: capunit_module if name == "e7awghal.capunit" else None,
    )

    patch.apply_e7awghal_classification_buffer_patch()

    capunit = capunit_module.CapUnitSimplified()
    capunit._current_param = _FakeCaptureParam(
        classification_enable=True,
        datasize_in_sample=5,
    )

    result = capunit._allocate_read_buffer(tag="classification")

    assert capunit._mm.calls == [(2, 64, {"tag": "classification"})]
    assert isinstance(result.bufsize, int)
    assert capunit.original_calls == 0


def test_apply_patch_preserves_original_nonclassification_path(monkeypatch) -> None:
    """Given standard IQ capture, when buffer allocates, then original path remains in use."""
    capunit_module = SimpleNamespace(
        CapUnitSimplified=_FakeCapUnitSimplified,
        _CAP_MINIMUM_ALIGN=64,
    )
    monkeypatch.setattr(patch, "_is_quelware_0_10_or_later", lambda: True)
    monkeypatch.setattr(
        patch.importlib,
        "import_module",
        lambda name: capunit_module if name == "e7awghal.capunit" else None,
    )

    patch.apply_e7awghal_classification_buffer_patch()

    capunit = capunit_module.CapUnitSimplified()
    capunit._current_param = _FakeCaptureParam(
        classification_enable=False,
        datasize_in_sample=5,
    )

    result = capunit._allocate_read_buffer(tag="iq")

    assert capunit._mm.calls == [(40, 64, {"tag": "iq"})]
    assert result.bufsize == 40
    assert capunit.original_calls == 1
