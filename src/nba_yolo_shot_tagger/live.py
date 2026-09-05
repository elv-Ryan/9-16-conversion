import struct
from typing import Protocol
from loguru import logger
import requests

class VerticalSink(Protocol):

    def publish(self, x_vals: list[float]) -> None:
        ...

def encode_x(value: float) -> bytes:
    """Encode a float in [0.0, 1.0] as a 4-byte little-endian fixed-point value."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"value must be in [0.0, 1.0], got {value}")
    return struct.pack("<I", int(value * 10_000 + 0.5))

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

    def publish(self, x_vals: list[float]) -> None:
        if not x_vals:
            return

        self.x_vals += x_vals

        url = f"{self.url}/q/{self.q}/call/live/data_streams/{self.stream}"

        blob = b"".join(encode_x(x) for x in self.x_vals)

        try:
            resp = requests.post(url, params={"authorization": self.tok}, data=blob, timeout=30)
            resp.raise_for_status()
        except Exception:
            logger.opt(exception=True).error(f"Post to fabric failed for url={url}")
            return

        self.x_vals.clear()