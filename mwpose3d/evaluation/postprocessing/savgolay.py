from collections import deque
import numpy as np
import torch
from .base import BasePostProcessing
from .base import POSTPROCESSING
from mwcore.utils.smoothing.savgol_filter import savgol_filter, SavGolayConfig, SavGolayPadding


@POSTPROCESSING.register_module()
class SavGolayFilter(BasePostProcessing):
    def __init__(self, 
                 window_length: int,
                 polyorder: int,
                 deriv: int = 0,
                 delta: float = 1.0,
                 mode: SavGolayPadding = 'reflect',
                 cval: float = 0.0,
                 time_axis: int = 0):
        super().__init__(online_mode=False)  # still fine

        self.config = SavGolayConfig(
            window_length=window_length,
            polyorder=polyorder,
            deriv=deriv,
            delta=delta,
            mode=mode,
            cval=cval
        )
        self.time_axis = int(time_axis)
        self._W = int(window_length)
        self._buffer = deque(maxlen=self._W)

    def reset(self):
        self._buffer.clear()

    def _stack_buffer(self) -> np.ndarray:
        """Stack frames in the buffer along `time_axis`."""
        # expand each frame with a singleton time axis, then concat along that axis
        return np.concatenate(
            [np.expand_dims(f, axis=self.time_axis) for f in self._buffer],
            axis=self.time_axis
        )

    def transform(self, datasample):
        orig = datasample.pred
        is_torch = isinstance(orig, torch.Tensor)
        device = orig.device if is_torch else None
        dtype = orig.dtype if is_torch else orig.dtype

        # Convert one incoming frame to numpy (whatever its shape is)
        frame_np = orig.detach().cpu().numpy() if is_torch else np.asarray(orig)
        
        # Push newest frame
        self._buffer.append(frame_np)

        # ── Pass-through until the window is full ───────────────────────────────
        if len(self._buffer) < self._W:
            # Not enough history for a causal SG window -> return the input
            print(f"Buffer length {len(self._buffer)} < window length {self._W}, returning original data.")
            datasample.pred = orig
            return datasample
        # ────────────────────────────────────────────────────────────────────────
        # Stack buffered frames along time_axis (length == self._W)
        x_win = self._stack_buffer()
        # Causal, zero-lag: evaluate at right edge; left-only padding
        y_all = savgol_filter(
            x_win,
            axis=self.time_axis,
            config=self.config,
            eval_index=self._W - 1,
            pad=(self._W - 1, 0),
        )
        # Take the "now" sample (last index on time_axis)
        y_now = np.take(y_all, indices=-1, axis=self.time_axis)

        # Back to the original container/dtype
        if is_torch:
            datasample.pred = torch.as_tensor(y_now, device=device, dtype=dtype)
        else:
            datasample.pred = y_now.astype(dtype, copy=False)

        return datasample

