from __future__ import annotations

from typing import List, Optional

import numpy as np


from .config import SHOT_DETECTION_LOOKAHEAD_FRAMES

# TransNetV2 slides 100-frame windows and keeps the middle 50 predictions, so
# every frame it commits to normally has ~25 real frames of context on each
# side. Where it does not have them it pads by duplicating the edge frame,
# which reads as a freeze and produces a spurious cut. Keeping this much real
# video on each side of the decision point is what avoids that. Left context
# is free, so it is more generous than the lookahead, which costs latency.
LOOKAHEAD_FRAMES = SHOT_DETECTION_LOOKAHEAD_FRAMES
CONTEXT_FRAMES = 50


class StreamShotDetector:
    """Shot-cut detection over a stream delivered as consecutive files.

    Run per file, TransNetV2 sees a duplicated-frame edge at both ends of every
    file and reports cuts there that are not in the video. This carries frames
    across the joins instead: each file is appended to a rolling buffer that
    still holds the tail of its predecessor, and a cut is only committed once
    there are ``LOOKAHEAD_FRAMES`` of real video after it. Frames past that
    point stay in the buffer and are re-examined on the next call, when the
    following file has supplied the missing context.

    Cuts are returned relative to the start of the file just pushed, so a cut
    in the previous file's deferred tail comes back negative.
    """

    def __init__(
        self,
        transnet_path: str,
        device: Optional[str] = None,
        lookahead_frames: int = LOOKAHEAD_FRAMES,
        context_frames: int = CONTEXT_FRAMES,
    ) -> None:
        # Imported here so the module can be loaded without torch present.
        import torch
        from shot.transnet import TransNetV2

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transnet = TransNetV2(transnet_path, device=device)
        self.lookahead_frames = lookahead_frames
        self.context_frames = context_frames

        self._buffer: Optional[np.ndarray] = None
        self._buffer_start = 0  # absolute stream frame index of _buffer[0]
        self._consumed = 0  # absolute frames decoded so far
        self._decided_through = 0  # absolute frame; cuts before this are committed
        self._file_start = 0  # absolute frame index of the file last pushed

    def push(self, path: str) -> List[int]:
        """Add a file to the stream, returning cuts relative to its start."""
        frames = self.transnet.decode_video(path)
        self._file_start = self._consumed
        if self._buffer is None or len(self._buffer) == 0:
            self._buffer = frames
            self._buffer_start = self._consumed
        else:
            self._buffer = np.concatenate([self._buffer, frames])
        self._consumed += len(frames)
        return self._commit(self._consumed - self.lookahead_frames)

    def flush(self) -> List[int]:
        """Commit whatever is left; there is no more video to wait for."""
        return self._commit(self._consumed)

    def _commit(self, horizon: int) -> List[int]:
        """Commit cuts up to ``horizon``, keeping the rest of the buffer."""
        if self._buffer is None or len(self._buffer) == 0:
            return []
        cuts = []
        for index in self.transnet.predict_frames_shots(self._buffer):
            # The transition ends on this frame, so the next shot starts after it.
            cut = self._buffer_start + index + 1
            if self._decided_through < cut <= horizon:
                cuts.append(cut)
        self._decided_through = max(self._decided_through, horizon)

        # Keep everything still undecided, plus real left context for it.
        keep_from = max(0, horizon - self.context_frames - self._buffer_start)
        if keep_from:
            self._buffer = self._buffer[keep_from:]
            self._buffer_start += keep_from
        return [cut - self._file_start for cut in sorted(set(cuts))]
