import numpy as np
import pytest

from twokwatcher.config import Config, Region


def test_region_scales_to_any_resolution():
    r = Region("x", x=0.5, y=0.25, w=0.1, h=0.2)
    assert r.to_pixels(1920, 1080) == (960, 270, 192, 216)
    assert r.to_pixels(1280, 720) == (640, 180, 128, 144)


def test_crop_clamps_instead_of_throwing():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # A region running off the right edge should degrade, not raise.
    crop = Region("edge", x=0.9, y=0.9, w=0.5, h=0.5).crop(frame)
    assert crop.shape[:2] == (10, 10)


def test_example_config_loads():
    config = Config.load()
    assert "scoreboard" in config.regions
    assert config.capture["sample_fps"] > 0


def test_unknown_region_names_the_fix():
    with pytest.raises(KeyError, match="calibrate"):
        Config.load().region("does_not_exist")


def test_roundtrip(tmp_path):
    config = Config.load()
    path = config.save(tmp_path / "regions.yaml")
    assert Config.load(path).regions.keys() == config.regions.keys()
