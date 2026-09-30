import math
import struct
from typing import Protocol
from loguru import logger
import requests

class VerticalSink(Protocol):

    def publish(self, x_vals: list[float]) -> None:
        ...

def encode_x(value: float) -> bytes:
    """Encode a float in [0.0, 1.0] as a 4-byte little-endian fixed-point value.

    Out-of-range clamps rather than raising. This runs inside the per-file
    commit, so raising here does not lose one sample -- it aborts the whole
    segment and turns it into an Error message. A crop centre a hair outside
    the legal interval is worth clamping silently; losing the segment is not.
    Anything non-finite falls back to centre, the same answer the trajectory
    itself gives when it has nothing to go on.
    """
    if not math.isfinite(value):
        value = 0.5
    clamped = min(1.0, max(0.0, float(value)))
    return struct.pack("<I", int(clamped * 10_000 + 0.5))

class NoopSink(VerticalSink):
    def publish(self, x_vals: list[float]) -> None:
        return

class FileSink(VerticalSink):
    """writes x vals to a file"""

    def __init__(self, out_path: str) -> None:
        self.out_path = out_path

    def publish(self, x_vals: list[float]) -> None:
        with open(self.out_path, 'ab') as fout:
            for x in x_vals:
                fout.write(encode_x(x))

class FabricSink(VerticalSink):
    """writes x vals to the fabric"""

    def __init__(
        self, 
        live_q: str, 
        base_url: str, 
        tok: str,
        data_stream: str
    ):
        self.q = live_q
        self.url = base_url
        self.tok = tok
        self.stream = data_stream
        self.x_vals = []
        self._initialized = False

    def publish(self, x_vals: list[float]) -> None:
        if not x_vals:
            return

        self.x_vals += x_vals

        try:
            self._create_stream_if_not_exists()
        except Exception:
            logger.opt(exception=True).error(f"Failed to create new stream")
            return

        url = f"{self.url}/q/{self.q}/call/live/data_streams/{self.stream}"

        blob = b"".join(encode_x(x) for x in self.x_vals)

        try:
            resp = requests.post(url, params={"authorization": self.tok}, data=blob, timeout=30)
            resp.raise_for_status()
        except Exception:
            logger.opt(exception=True).error(f"Post to fabric failed for url={url}")
            return

        self.x_vals.clear()

    def _create_stream_if_not_exists(self) -> None:
        if self._initialized:
            return

        url = f"{self.url}/q/{self.q}/call/live/data_streams"

        resp = requests.post(url, params={"authorization": self.tok}, json={"name": self.stream}, timeout=30)
        if resp.status_code == 409:
            # already set
            self._initialized = True
            return
        
        resp.raise_for_status()

        self._initialized = True