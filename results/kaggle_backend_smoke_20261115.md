# Kaggle backend smoke evidence — supplied 15 November 2026

The notebook export supplied for the competition contains these successful outputs:

```text
Serving bundle: /kaggle/input/datasets/penguin069/behaviorsense-rgb-hybrid-serving
Verified runtime files: 49
Model: rgb-hybrid-20261015-t4-fp32-v1
GPU 0 smoke passed: Tesla T4 float32
GPU 1 smoke passed: Tesla T4 float32
Ready for two simultaneous clips. Reports run through your local backend.
```

The detector inputs recorded in the export are `weights/rtmo-l.onnx` and `rtdetr-l.onnx`
from `krishnagupta02468/behavioursense-ww`. The final cell also printed the command
for connecting the local bridge to its Cloudflare tunnel.

These outputs demonstrate successful model forwards and backend startup on two T4 GPUs.
They are not an accuracy benchmark or proof of medication adherence. The export was
provided on 15 November; its output does not record a separate execution timestamp.

The [published notebook](../notebooks/notebookef04f4f9bb%20%281%29.ipynb) keeps the
executable code and removes saved cell output so each user generates a fresh tunnel URL.
The historical [RGB benchmark](rgb_improvement_20261015.md) remains separate.
