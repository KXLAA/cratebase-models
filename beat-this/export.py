"""Exports Beat This's small checkpoint to ONNX and checks the file against PyTorch.

    python3.12 -m venv venv
    venv/bin/pip install -r requirements.txt
    venv/bin/python export.py

Writes beat-this-small.onnx beside this script.
"""

import numpy as np
import onnxruntime as ort
import torch
from beat_this.inference import load_model

CHECKPOINT = "small0"
OUTPUT = "beat-this-small.onnx"
MELS = 128
# The length the model was trained on: 30 seconds at 50 frames a second.
TRACED_FRAMES = 1500
# The file must answer every window length the same as PyTorch does, not only the traced one.
CHECKED_FRAMES = (300, 1500, 2000)
TOLERANCE = 1e-3


class Logits(torch.nn.Module):
    """The model with its two outputs as a tuple, which is what ONNX can name."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, spectrogram):
        out = self.model(spectrogram)
        return out["beat"], out["downbeat"]


def main():
    torch.manual_seed(0)
    model = Logits(load_model(CHECKPOINT, "cpu")).eval()

    torch.onnx.export(
        model,
        (torch.rand(1, TRACED_FRAMES, MELS) * 6,),
        OUTPUT,
        input_names=["spectrogram"],
        output_names=["beat", "downbeat"],
        dynamic_axes={"spectrogram": {1: "time"}, "beat": {1: "time"}, "downbeat": {1: "time"}},
        opset_version=17,
        dynamo=False,
    )

    session = ort.InferenceSession(OUTPUT, providers=["CPUExecutionProvider"])
    for frames in CHECKED_FRAMES:
        # A log-mel spectrogram's values lie between 0 and about 7.
        spectrogram = torch.rand(1, frames, MELS) * 6
        with torch.no_grad():
            beat, downbeat = model(spectrogram)
        exported = session.run(None, {"spectrogram": spectrogram.numpy()})
        worst = max(
            np.abs(exported[0] - beat.numpy()).max(),
            np.abs(exported[1] - downbeat.numpy()).max(),
        )
        print(f"{frames} frames: differs from PyTorch by at most {worst:.6f}")
        if worst > TOLERANCE:
            raise SystemExit(f"{OUTPUT} is wrong at {frames} frames")


if __name__ == "__main__":
    main()
