# cratebase-models

The machine-learning models [Cratebase](https://github.com/KXLAA/cratebase) ships, as ONNX files, with the scripts that made them.

## Releases

Each release is tagged `<model>-<build>` and holds the model's `.onnx` file and a `SHA256SUMS`. Cratebase pins a release by its checksum.

## Models

### beat-this-small

Finds beats and downbeats. It is the small checkpoint (`small0`, 2.1 million parameters) of [Beat This!](https://github.com/CPJKU/beat_this) by Foscarin, Schlüter and Widmer (ISMIR 2024), exported unchanged.

| | Name | Shape | |
| --- | --- | --- | --- |
| Input | `spectrogram` | `[batch, time, 128]` float32 | The log-mel spectrogram Beat This is trained on |
| Output | `beat` | `[batch, time]` float32 | A logit per frame: over 0, a beat is more likely than not |
| Output | `downbeat` | `[batch, time]` float32 | The same for the first beat of a bar |

The model does not include its spectrogram. The caller makes it:

- mono audio at 22,050 Hz;
- frames of 1,024 samples under a Hann window, every 441 samples (50 frames a second), centred with reflected edges;
- magnitudes divided by the square root of 1,024;
- 128 Slaney mel bands from 30 Hz to 11 kHz, not normalised;
- `log1p(1000 * x)`.

`time` may be any length. The model was trained on 1,500 frames (30 seconds), and `export.py` checks the file against PyTorch at 300, 1,500 and 2,000 frames.

To build it again:

```sh
cd beat-this
python3.12 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python export.py
```

## Licences

The scripts here are MIT licensed (`LICENSE`). Each model keeps the licence of the weights it was exported from: Beat This's code and weights are MIT licensed (`beat-this/LICENSE`).
