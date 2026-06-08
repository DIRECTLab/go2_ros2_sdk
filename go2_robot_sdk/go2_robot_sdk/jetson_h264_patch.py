import asyncio
import threading
import fractions
import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
Gst.init(None)

from av import VideoFrame


class JetsonH264Decoder:
    def __init__(self):
        self._loop = None
        self._output_q = None

        pipeline_str = (
            "appsrc name=src stream-type=0 format=time is-live=true "
            "caps=video/x-h264,stream-format=byte-stream,alignment=au ! "
            "h264parse ! "
            "nvv4l2decoder ! "
            "nvvidconv ! "
            "video/x-raw,format=BGRx ! "
            "appsink name=sink emit-signals=true max-buffers=2 drop=true"
        )
        self._pipeline = Gst.parse_launch(pipeline_str)
        self._src = self._pipeline.get_by_name("src")
        self._sink = self._pipeline.get_by_name("sink")
        self._sink.connect("new-sample", self._on_sample)
        self._pipeline.set_state(Gst.State.PLAYING)
        self._pts = 0

    def set_output(self, loop, output_q):
        """Call this once before decode() to enable async push."""
        self._loop = loop
        self._output_q = output_q

    def _on_sample(self, sink):
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        caps = sample.get_caps()
        s = caps.get_structure(0)
        width = s.get_value("width")
        height = s.get_value("height")
        ok, map_info = buf.map(Gst.MapFlags.READ)
        if ok:
            arr = np.frombuffer(map_info.data, dtype=np.uint8)
            arr = arr.reshape((height, width, 4))[:, :, :3].copy()
            frame = VideoFrame.from_ndarray(arr, format="bgr24")
            frame.pts = self._pts
            frame.time_base = fractions.Fraction(1, 90000)
            buf.unmap(map_info)
            # push directly to aiortc's output queue from GStreamer thread
            if self._loop and self._output_q:
                asyncio.run_coroutine_threadsafe(
                    self._output_q.put(frame), self._loop
                )
        return Gst.FlowReturn.OK

    def decode(self, encoded_frame) -> list:
        self._pts = encoded_frame.timestamp
        buf = Gst.Buffer.new_wrapped(bytes(encoded_frame.data))
        buf.pts = encoded_frame.timestamp
        self._src.emit("push-buffer", buf)
        return []  # never block; frames pushed async from _on_sample

    def __del__(self):
        if hasattr(self, '_pipeline'):
            self._pipeline.set_state(Gst.State.NULL)
