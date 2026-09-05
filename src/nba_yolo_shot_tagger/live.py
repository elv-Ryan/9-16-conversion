import struct
from typing import Protocol

class VerticalSink(Protocol):

    def publish(self, x_vals: list[float]) -> None:
        ...

def encode_x(value: float) -> bytes:
    """Encode a float in [0.0, 1.0] as a 4-byte little-endian fixed-point value."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"value must be in [0.0, 1.0], got {value}")
    return struct.pack("<I", int(value * 10_000 + 0.5))

class FileSink(VerticalSink):

    def __init__(self, out_path: str) -> None:
        self.out_path = out_path

    def publish(self, x_vals: list[float]) -> None:
        with open(self.out_path, 'ab') as fout:
            for x in x_vals:
                fout.write(encode_x(x))
