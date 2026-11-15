# RGB hybrid serving on T4 ×2

Use the supplied [`notebookef04f4f9bb (1).ipynb`](../notebooks/notebookef04f4f9bb%20%281%29.ipynb) to obtain the Kaggle backend link for the current demo. The model inputs are public, and the supplied export records successful startup on both T4 GPUs.

## Inputs to attach

| Input | How to obtain it | Contents used |
|---|---|---|
| **[`penguin069/behaviorsense-rgb-hybrid-serving`](https://www.kaggle.com/datasets/penguin069/behaviorsense-rgb-hybrid-serving)** | Add this public model bundle to the Kaggle notebook | GPU runtime source; context seed 2, temporal ASL seed 1 and RGB head; frozen thresholds and normalization; Swin3D-T; OSNet for optional enrollment |
| **[`krishnagupta02468/behavioursense-ww`](https://www.kaggle.com/datasets/krishnagupta02468/behavioursense-ww)** | Add this public weights dataset | `weights/rtmo-l.onnx` and the validated `rtdetr-l.onnx` |

The serving dataset has `hybrid_manifest.json`, `source_manifest.json` and `runtime_src/` at its root. The notebook supports an explicit `BUNDLE_DIR` if multiple bundles are attached. Its GPU code comes from this versioned model bundle; the GitHub checkout supplies the local bridge and web app. Pretrained weights retain their upstream licenses.

No Qwen model dataset, original Toyota ADL runs, separate code dataset, Toyota videos, annotations, pose shards, RGB caches, or training/evaluation outputs are required. Historical training and benchmark files remain unchanged.

## Run

1. Import the supplied notebook into Kaggle and attach the two public inputs above.
2. In notebook Settings select **GPU T4 x2** and **Internet On**.
3. Run the cells in order. They verify the packaged code and model hashes, install Torch 2.11 / torchvision 0.26 with CUDA 12.8 and ONNX Runtime GPU 1.26, and test all required model forwards on each GPU.
4. Keep the last cell running. It prints `python web/local_backend.py --kaggle https://...trycloudflare.com` for your Windows terminal. If `BS_TOKEN` is set, also pass the matching `--token` to the local command.
5. Keep your OpenRouter keys on the local machine and connect the web page to `http://127.0.0.1:8899`.

The Kaggle service supplies stages 1–2. The existing local bridge handles stages 3–4, translation and verification. It detects the service's `chunks-v1` upload support automatically. Each active upload owns one GPU and one subprocess; at most two process concurrently, with two waiting reservations. Model allocations disappear when the upload subprocess exits, and disconnects terminate that upload's worker before its GPU is reused.

Default limits are 100 MB per file and 36,000 sampled frames (30 minutes at 20 FPS). Set `BS_MAX_UPLOAD_MB` and `BS_MAX_FRAMES` before startup to change them. Sources below 20 FPS are refused. Windows contain 30 samples with a 15-sample stride. No end padding or gap filling creates extra windows.

## Model and portability

The classifier preserves the saved probability blend: 37.5% context seed 2, 37.5% temporal ASL seed 1 and 25% RGB. It uses the frozen class thresholds, ±30-frame RGB pair support and central-TV condition. All other class flags are preserved; no single-label softmax or Viterbi decoder is applied. The seven classes disabled by the frozen thresholds stay disabled, including the unscored fall classes.

The model's temporal context can look 4.5 seconds ahead, and its pair rule uses 1.5 seconds of centered support. This is offline inference. Per-window decisions occupy disjoint 0.75-second center bins, preserving gaps and avoiding simultaneous activities created solely by overlapping output intervals. The tracker supplies each target person in the primary skeleton slot and a co-present person in the secondary slot.

The original benchmark used BF16 RGB extraction on a different GPU. T4 has no native BF16 support, so this separate serving adapter uses **FP32** and retains float16 feature storage before the RGB head. It also maps uploaded-video tracks to the primary-person inputs. These choices preserve weights, image preprocessing, model architectures and thresholds, but do not establish identical predictions or benchmark accuracy on T4. The original **69.21% mAP / 28.12% pair F1** remains a historical result; its pair quality gate failed. `/health` and returned model metadata identify the FP32 port and do not claim a new verified benchmark.

Local checks compare all three loaded heads and the RGB control features against the existing inference implementations, exercise temporal and track boundaries, test concurrent uploads and cancellation, and run a synthetic video through the complete response adapter. The supplied notebook export records successful model-forward smoke checks on both Tesla T4 GPUs; each new session repeats those checks. [Recorded Kaggle evidence](../results/kaggle_backend_smoke_20261115.md).

The earlier local serving checks passed **42 tests and 19 subtests**. A separate replay reproduced every saved class decision across **63,220 validation windows**; the maximum probability difference was 2.98e-8. That replay tests the blend and decoder using saved component logits, not the FP32 RGB encoder.

For development, the original packaging utility can rebuild a local model bundle when the original checkpoint exports are available:

```powershell
python scripts/package_rgb_hybrid_serving.py
python -m pytest tests/test_rgb_hybrid_serving.py -q
```

The packaging command performs no network requests or Kaggle changes. Rebuilding requires the original local checkpoint exports; ordinary demo users can attach the public datasets above. The older `08_serve_rgb_hybrid_t4x2.ipynb` is retained as a development template. The supplied notebook is the primary run path.
