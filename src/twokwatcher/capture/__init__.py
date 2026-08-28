"""Frame sources. Everything downstream sees frames, never a capture device."""

from .source import Frame, FrameSource, open_source
from .videofile import VideoFileSource
from .virtualcam import VirtualCamSource, list_devices

__all__ = [
    "Frame",
    "FrameSource",
    "VideoFileSource",
    "VirtualCamSource",
    "list_devices",
    "open_source",
]
