"""Regression tests for platform-specific native dependency constraints."""

from importlib.metadata import metadata


def test_linux_pedalboard_wheels_exclude_known_sigill_releases():
    requirements = metadata("amt-augmentor").get_all("Requires-Dist") or []
    pedalboard = [
        requirement
        for requirement in requirements
        if requirement.startswith("pedalboard")
    ]

    linux = [
        requirement
        for requirement in pedalboard
        if 'sys_platform == "linux"' in requirement
    ]
    non_linux = [
        requirement
        for requirement in pedalboard
        if 'sys_platform != "linux"' in requirement
    ]

    assert len(linux) == 1
    assert ">=0.7.3" in linux[0]
    assert "<0.9.21" in linux[0]
    assert len(non_linux) == 1
    assert ">=0.7.3" in non_linux[0]
    assert "<1.0.0" in non_linux[0]
