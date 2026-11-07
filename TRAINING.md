> Historical research guide. The camera dashboard uses the root README and needs no training. Historical datasets, earlier model variants and iOS app workspace are separate assets not bundled in this checkout.

# Turtle detection with YOLO26 nano

For the current audited training workflow, use [Challenge experiments, 2026-10-19](#challenge-experiments-2026-10-19). The sections before it document the earlier RGB experiment and existing iOS integration.

For new paired RGB/mono/depth recordings from the connected OAK-D Pro, see [OAK_CAPTURE.md](OAK_CAPTURE.md). The recorder uses a separate camera environment and does not change this RGB training pipeline.

The dataset comes from `turtle-hero.ndjson`: 199 training images and 50 validation images, with one class (`0: turtle`). The original split is preserved. Images are downloaded from the export, checked against their recorded dimensions, and converted into YOLO detection labels in `datasets/turtle-hero/`. Preparation rejects invalid labels and exact duplicate image content across splits.

## Train

```sh
uv sync
.venv/bin/python -u train.py
```

The default run fine-tunes pretrained `yolo26n.pt` at 640 pixels, batch size 8, for up to 100 epochs, with early stopping after 20 epochs without improvement. It selects the Mac GPU (MPS) when available, otherwise the CPU. Use `--device cpu`, `--epochs`, or `--batch` to override these settings. Repeated runs get separate output folders.

```sh
.venv/bin/python train.py --prepare-only
.venv/bin/python train.py --resume runs/detect/turtle_yolo26n/weights/last.pt
```

## Outputs

- `runs/detect/turtle_yolo26n/weights/best.pt`: best validation checkpoint.
- `runs/detect/turtle_yolo26n/weights/last.pt`: final/latest checkpoint.
- `runs/detect/turtle_yolo26n/results.csv`: per-epoch training and validation metrics.
- `runs/detect/turtle_yolo26n/args.yaml`: actual training configuration.
- `training.log`: console output from the initial training run.

Use the trained detector:

```python
from ultralytics import YOLO

model = YOLO('runs/detect/turtle_yolo26n/weights/best.pt')
model.predict(source='path/to/image.jpg', imgsz=640, conf=0.016, save=True)
```

Validation metrics measure performance on the 50-image validation split used for model selection; they are not an independent test-set estimate. This export contains turtle images and does not establish false-positive performance on turtle-free scenes.

## Completed run

Training finished after 21 epochs (early stopping, best checkpoint from epoch 1). A separate validation pass at batch size 8 measured the best checkpoint at 72.9% mAP50 and 55.4% mAP50–95. Results differ slightly from the end-of-training validation pass, which used a different batch size.

Confidence scores for this checkpoint are unusually low. The validation F1 curve peaks near `conf=0.016`, with approximately 80.2% precision and 71.3% recall at IoU 0.5. Use that threshold as a starting point rather than the usual 0.25; it was selected on the validation set and requires checking on new images. The framework's smoothed confidence curve obscures this narrow peak, explaining its very low reported default operating-point precision.

`runs/detect/turtle_yolo26n/evaluation.json` compares both saved checkpoints. `runs/predict/best/` contains predictions for all 50 validation images. Rerun these checks with `.venv/bin/python evaluate.py` (requires MPS). This model is an initial small-dataset detector, with no independent test-set evaluation.

## Refined inference threshold

An exact confidence sweep using single-image inference on the 50 validation images selected approximately `conf=0.016388704` for both maximum F1 and maximum F2. At IoU 0.5 this produced 35 true positives, 1 false positive, and 16 false negatives (97.2% precision, 68.6% recall). These are validation-tuned results, not independent test estimates. The earlier 0.016 threshold produced 35 true positives and 7 false positives under this evaluation protocol.

The five new iNaturalist predictions were regenerated in `runs/new_inaturalist/annotated/`; four images have detections and one remains missed. `predict_new.py` now reads the selected threshold from `runs/new_inaturalist/confidence_tuning.json`. Reproduce tuning with `.venv/bin/python tune_confidence.py`.

## iOS app integration

Open `ios-turtle/YOLOiOSApp/YOLOiOSApp.xcodeproj`. This is the `turtle-detector` feature worktree of the parent `yolo-ios-app` repository. The original checkout remains untouched.

The bundled `YOLOiOSApp/Models/Detect/turtle.mlpackage` is selected in the Detect task's Nano slot. Keep the Core AI Models setting off. The app applies `0.016388704` confidence after loading, including on an external display; confidence controls retain more than two decimal places. Select your signing team and physical iPhone in Xcode to run it.

Export with `.venv-coreml/bin/python export_coreml.py`. The final export uses 640×640 input, FP16 and **nms=None**, retaining the raw detection head used by the checkpoint. The app decodes it and runs NMS in Swift. In Ultralytics 8.4.157, `nms=False` selects the other detection branch and materially reduced validation accuracy for this checkpoint; do not substitute that setting.

Validation: the final Core ML export scored 71.35% mAP50 and 53.90% mAP50–95 on the 50 validation images, compared with 72.04% and 54.18% for PyTorch with the same square input. It detected turtles in the same four of five new iNaturalist photos. See `runs/evaluate/coreml_report.json` and `runs/predict/coreml-2/`. Live camera inference and device-specific confidence still require testing on an iPhone.

Core ML packages are gitignored by the iOS repository; the bundled turtle model is present locally and must be copied separately when sharing the source changes.

## Challenge experiments, 2026-10-19

These experiments address the incremental RGB/P2 stages of `instructions.md`. They do not establish construction-site safety or a trained multimodal system. The earlier run and bundled iOS model above remain unchanged.

`train_challenge.py` uses the existing local photos, with no dataset download or cloud logging. A visually confirmed near-duplicate, training image `6516879.jpg`, is excluded because it repeats validation image `10066100.jpg`. The corrected split has 198 training images (202 boxes) and 50 validation images (51 boxes). Original images and labels are preserved; the new manifest, hashes, exclusion record, and split lists live in `datasets/turtle-challenge-v1/`.

Both runs use detection-pretrained YOLO26n weights, 640-pixel input, AdamW at 0.0003, batch 8, seed 42, up to 60 epochs, and patience 15. Configuration and source hashes are saved with each run. MPS has nondeterministic reductions, so a seed does not guarantee bitwise repeatability. Dependencies are pinned in `requirements-training.txt`; the complete environment snapshot is in `evidence/training-20261019/environment.lock.txt`.

The P2 architecture comes from the installed, pinned Ultralytics 8.4.157 configuration. `p2_transfer.json` records tensor-by-tensor transfer using corresponding layer and stride identities, accounting for shifted neck/head indices. It retains the native four-scale detection head and progressive one-to-many/one-to-one loss with small-box task-aligned assignment. Inference uses the one-to-many branch followed by NMS.

Reproduce with new run names (existing runs are protected):

```sh
.venv/bin/python train_challenge.py --variant rgb --name rgb_repeat
.venv/bin/python train_challenge.py --variant p2 --name p2_repeat
.venv/bin/python evaluate_challenge.py runs/challenge/rgb_repeat/weights/best.pt runs/challenge/p2_repeat/weights/best.pt --output runs/challenge/comparison_repeat
.venv/bin/python select_challenge.py --comparison runs/challenge/comparison_repeat/comparison.json --output models/challenge-repeat
```

Evaluation uses square 640×640 letterboxing, one image per batch, one-to-many predictions, confidence floor 0.001, NMS IoU 0.7, and COCO metrics with up to 100 detections. Model selection uses validation AP50–95; threshold selection maximizes validation F2. Exact threshold sweeps group equal scores and use one-to-one IoU ≥0.5 matching. Image bootstrap intervals are conditional on the selected threshold and this dataset; they do not capture selection bias or unknown encounter correlation.

The dataset contains no labeled turtle-free scenes, only two small validation turtles at 640 input size, and no encounter/site/species/occlusion metadata. Original-pixel COCO small-object counts differ from the counts after resizing; both are reported. Validation is used for model selection and is not an independent field test. Source licenses and individual/observation groups are absent from the export and remain unknown in the manifest.

Both runs completed: RGB stopped at epoch 48 (best epoch 33), and P2 stopped at epoch 47 (best epoch 32). The fixed evaluation measured:

| Model | AP50–95 | AP50 | F2-selected precision | F2-selected recall | TP / FP / FN | Host median / p95 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Stock RGB | 54.93% | 81.72% | 80.39% | 80.39% | 41 / 10 / 10 | 17.95 / 25.96 ms |
| RGB + P2 | 55.07% | 84.40% | 66.67% | 86.27% | 44 / 22 / 7 | 20.59 / 23.07 ms |

P2 was selected by the predefined AP50–95 criterion, with only a 0.15 percentage-point lead. This does not establish a reliable architecture advantage. The selected threshold is `0.0724431499838829`; it prioritizes recall using validation F2 and increases false detections. Both models missed both small targets at 640 input. P2 matched 1/3 medium and 43/46 large targets at its selected threshold. Its conditional image-bootstrap 95% intervals are 54.4–80.4% precision and 74.5–96.0% recall; selection bias and encounter correlation are not included. There are no negative-only scenes to estimate false alerts per hour.

Host timings cover file read, preprocessing, inference, NMS, and CPU transfer on this Mac, excluding the first three images. They are not OAK or iPhone performance. The selected checkpoint is 5,558,459 bytes and has 2,515,976 unfused parameters. Full reports, predictions, and `comparison.png` are in `runs/challenge/comparison_20261019/`.

The selected model and comparison are packaged under `models/challenge-20261019/`; `model.json` contains the input/output conventions and selected threshold. Run local image inference or saved OAK replay with:

```sh
.venv/bin/python predict_challenge.py path/to/image.jpg --output runs/challenge/my-image
.venv/bin/python predict_challenge.py recordings/oak/20261016T205218_98151fbf --output runs/challenge/my-replay
.venv-coreml/bin/python export_challenge.py
```

The prediction script reads the packaged validation-selected threshold. RGB detections remain present when depth is missing. Optional distance association uses the central 50% of a box, a minimum of 20 measured pixels and 30% support, and a relative IQR limit of 0.2. These are configurable function parameters for an uncalibrated heuristic, not safety thresholds. Any reported distance is camera-axis depth and marked low quality; otherwise it is null/unknown. No tracking or machinery control is implemented.

The selected P2 model replayed the validated session `recordings/oak/20261016T205218_98151fbf` locally: 83 frames, 82 frames with predictions, and 110 detections. All 110 distances remained unknown. Predictions and sample overlays are in `runs/challenge/selected_oak_replay_20261019/`. Visual review found extra overlapping boxes as well as real turtle detections; these unreviewed recordings cannot provide accuracy measurements. Device-monotonic timestamps and clock semantics are retained in the prediction records.

The selected P2 ONNX and FP32 Core ML exports passed raw-output comparisons on three validation images. Maximum box-coordinate differences were 0.0044 pixels (ONNX) and 0.0070 pixels (Core ML), with maximum score differences below 0.000004. The raw shape is `[1, 5, 34000]`: decoded input-pixel `xywh` plus one turtle score, before host NMS. The four-scale head must not be decoded as a fixed 8,400-candidate stock head. See `models/challenge-20261019/export_report.json`; these checks do not constitute full exported-model task evaluation or physical-device testing.

### Measured-depth foundation and shell scan

`turtle_geometry.py` provides shared RGB/geometry letterboxing, finite-depth/validity handling, fixed log-depth normalization, inverse box mapping, and nearest-surface calibrated reprojection. `turtle_fusion.py` provides the depthwise encoder and zero-initialized residual gates at backbone strides 4/8/16. Quality downsampling uses mean support. Tests verify exact RGB fallback with zero geometry even after nonzero projection weights/biases, four detection scales, finite gradients, padding, holes, and reprojection. This module was initially untrained; the subsequent [fusion experiment](#trained-rgb-depth-fusion-2026-10-19) trained a separate checkpoint. It is not included in the original RGB package.

Both OAK sessions (159 samples total) have no reviewed boxes and are excluded from supervised training. There is no paired iPhone or annotated IR dataset available. Missing measured depth is never replaced with monocular predictions.

The candidate scan `assets/Scan.obj` has 997,943 vertices, 1,986,869 faces, UV coordinates, and a referenced `Scan.jpg` texture. Its physical units are unspecified. Its identity has not been confirmed by the user; the referenced texture is not plain white. A neutral inspection render shows a ragged/open region requiring review before synthesis. Inspection records, hashes, and previews are under `evidence/training-20261019/`. The original scan was not modified. No scan-derived examples were added to training, and no anatomical/scute restoration is claimed.

```sh
/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --python inspect_shell_blender.py -- --input assets/Scan.obj --output evidence/my-shell-preview
.venv/bin/python -m unittest discover -s tests/training -v
.venv-oak/bin/python -m unittest discover -s tests -v
```

The baseline FP16 Core ML candidate failed a conservative raw-coordinate parity check and is retained as unvalidated; FP32 passed on the three checked images. Export reports retain failed attempts. Export parity is separate from device performance: OAK RVC2 compilation, physical iPhone inference, full exported-model task metrics, and sensor latency have not been established for these new weights.

Validation completed with 14 training/preprocessing/fusion tests and 13 capture-contract tests passing. The report figure and prediction overlays were inspected. Logs are retained under `evidence/training-20261019/`.

Luxonis MCP `search_docs` and `get_doc_page` succeeded during the conversion preflight. The retrieved [ModelConverter documentation](https://docs.luxonis.com/software-v3/ai-inference/conversion/rvc-conversion/offline/modelconverter) is saved in `evidence/training-20261019/modelconverter-doc.md`. Local compilation requires the platform Docker toolchain. The installed Docker client could not connect to its daemon, and ModelConverter is not installed; no compiler was invoked and no RVC2 compatibility is claimed. See `rvc2-preflight.json` in the evidence directory. The next conversion step is to start Docker Desktop and install a pinned ModelConverter in a separate environment, then check the selected ONNX with explicit input preprocessing and four-scale output decoding. No model upload was performed.

Primary implementation references: [YOLO26 architectures and inference branches](https://docs.ultralytics.com/models/yolo26), [Ultralytics training](https://docs.ultralytics.com/modes/train), the installed 8.4.157 model/loss/assignment source, and the [COCO evaluator](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py).

## Small-scale fine-tuning follow-up, 2026-10-19

An exploratory RGB-only fine-tuning run initialized from the selected P2 checkpoint, used image-scale gains of 0.2–1.0 instead of the original 0.7–1.3, and lowered AdamW's initial learning rate to 0.0001. Other settings and the audited split were retained. This is a combined fine-tuning recipe comparison, not an isolated causal ablation. The scale bounds are supported by the pinned implementation and [official augmentation documentation](https://docs.ultralytics.com/guides/yolo-data-augmentation). Training batch overlays were inspected to verify the transformed boxes and images.

```sh
.venv/bin/python train_challenge.py --variant p2 --init-checkpoint models/challenge-20261019/turtle.pt --scale-range 0.2 1.0 --learning-rate 0.0001 --epochs 20 --patience 8 --name p2_small_repeat
```

The actual run `runs/challenge/p2_small_20261019/` stopped after nine epochs, selecting epoch one. Fixed-protocol evaluation on the same 50 validation images measured:

| Model | AP50–95 | AP50 | F2-selected precision | F2-selected recall | TP / FP / FN | Small recall at 640 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Original P2 | 55.07% | 84.40% | 66.67% | 86.27% | 44 / 22 / 7 | 0 / 2 |
| Small-scale fine-tune | 50.20% | 82.76% | 87.50% | 82.35% | 42 / 6 / 9 | 0 / 2 |

The original packaged model remains selected under the AP50–95 selection criterion. The fine-tune has fewer false detections at its separately tuned threshold, but reduced recall, lower AP, and no small-target gain. Its checkpoint is retained for inspection; it was not substituted into the deployed artifact directory or exported. Repeated selection on this validation set is exploratory and increases selection bias. Reports and the comparison figure are in `runs/challenge/comparison_small_20261019/`; logs and source snapshots are in `evidence/training-20261019-small/`.

Both real OAK sessions passed structural validation. The second session, `20261016T205810_f6a2ab4a`, has 54.67% mean valid depth versus 0.74% in the first. The original P2 model produced 93 proposals in the second recording; 80 received low-quality interior-depth estimates and 13 remained unknown. Proposed boxes include false detections, so these estimates are not verified animal ranges. The second recording is a more promising paired-depth training source after label review.

Two local review queues contain 24 selected frames each, with proposals retained for all 159 frames. [OAK_CAPTURE.md](OAK_CAPTURE.md#review-with-detector-proposals) gives the commands and review controls. Predictions remain separate from labels. Six new proposal/review tests passed, along with the 13 existing capture tests. The tests include rejecting mismatched images/views/sessions and requiring an explicit action before saving annotations. GUI key handling was checked with mocked display calls; a person has not completed the real annotation session. No scan-derived data, unreviewed OAK labels, uploads, or additional cloud services were used.

## Blender synthetic-data workflow, 2026-10-19

The subsequent Blender workflow uses `assets/Scan.obj` as the locally available shell asset. Its source hash is recorded and remains unchanged. The original physical scale is unknown: a **20 cm longest dimension is an explicit simulation assumption**, configurable with `--assumed-length-m`. Generated depth is never described as measured OAK or iPhone data.

The working mesh is reoriented, centered, decimated to 29,802 faces, cleaned of near-coincident/degenerate elements, and given recalculated normals. Large openings remain; there is no anatomical reconstruction. Three procedural materials provide mottled color and polygonal bump shading. They are stylized shell-proxy appearances, not validated species textures or restored scute geometry. This asset has no modeled head or limbs. No third-party texture photographs were downloaded or used.

The dataset contains 96 Cycles renders at 640×640: 84 shell-positive scenes and 12 shell-free synthetic negatives. Target sizes, viewpoints, soil/gravel/grass backgrounds, rocks, cover, and lighting vary. Some RGB training images receive mild Gaussian blur. The 84 synthetic boxes include 22 small, 43 medium, and 19 large targets under the input-pixel area convention. One scan and three procedural material presets do not establish biological diversity or realistic field coverage; the three materials are paired with the three environment styles in this first dataset.

Artifacts:

| Path | Contents |
| --- | --- |
| `datasets/blender-shell-v1/shell_workbench.blend` | Editable, packed Blender scene; F12 outputs go to `interactive-preview/` rather than the training images |
| `datasets/blender-shell-v1/asset.json` | Source hash, scale assumption, mesh cleanup, materials, Blender version, render settings |
| `datasets/blender-shell-v1/samples/` | RGB, visible-shell masks, ideal simulated camera-axis Z, validity, raw EXR position/index passes, and camera calibration per scene |
| `datasets/turtle-real-synthetic-v1/` | Audited 294-image training split: 198 original real images plus 96 synthetic images; unchanged 50-image real validation split |
| `datasets/turtle-real-synthetic-v1/geometry/` | OAK-like and iPhone-like corruption variants, validity, and their parameters |
| `datasets/turtle-real-synthetic-v1/contact-sheet.jpg` | Render and visible-box overview |
| `evidence/blender-20261019/geometry-qa.png` | RGB/mask/ideal-depth/degraded-depth comparison |

Bounding boxes come from the **visible** object-index mask, including actual geometry occlusion. Fully hidden or sub-eight-pixel shell fragments are excluded rather than assigned an ordinary positive label. Shell-free negative scenes are explicitly rendered with the shell hidden. All scan derivatives are confined to training. The real validation image list is byte-for-byte unchanged.

Blender world-position passes are transformed into the camera frame to obtain positive camera-axis Z, avoiding confusion with radial range. Intrinsics, extrinsics, units, image dimensions, and validity are saved. A rendered plane fixture measured exactly 2 m at every pixel and had maximum reprojection error below 0.001 pixels, checking depth convention and image orientation. All 96 renders passed file, shape, mask, visible-box, camera, and provenance validation. The training test suite passed 17 checks; the final three simulation-specific tests passed after the focal-length parameter update.

Depth corruption is an **uncalibrated stress test**: spatial downsampling, range limits, holes, edge rejection, quantization/noise, and one-pixel alignment perturbations. Unknown pixels remain zero/invalid. Quality uses binary validity, not a fabricated calibrated confidence probability. Profile parameters are saved for every sample. The controlled Blender comparison below uses RGB; the subsequent fusion experiment uses the depth maps.

Run the full workflow into fresh output folders:

```sh
.venv/bin/python run_blender_workflow.py --tag blender_next --scan assets/Scan.obj --assumed-length-m 0.20 --count 96 --epochs 20
```

The runner executes the geometry fixture, rendering, editable-scene packaging, dataset validation, matched real-only and real-plus-synthetic fine-tuning, real-image evaluation, checkpoint selection, and ONNX/Core ML export checks. It records commands and source snapshots and refuses existing output tags. New workflow artifacts go under `runs/blender-workflow/<tag>/`, training runs under `runs/challenge/`, and the selected package under `models/<tag>/`. It uses existing local environments and never uploads data or initiates paid training. If the unchanged original model remains best, its already-checked exports are copied with their provenance rather than rerun.

Primary Blender references: [render passes](https://docs.blender.org/manual/en/latest/render/layers/passes.html), [File Output API](https://docs.blender.org/api/5.0/bpy.types.CompositorNodeOutputFile.html), and the inspected Blender 5.2.2 RNA API. The installed version uses `Scene.compositing_node_group`, `format.media_type`, and the `Object Index` output name; older compositor examples are not copied blindly.

### Completed Blender experiment

Both fine-tuning runs initialized from the original selected P2 checkpoint with AdamW at 0.00005, batch 8, seed 42, the original augmentation recipe, a 20-epoch limit, and patience 8. Both stopped after nine epochs and selected epoch one. The recipe and epoch limits match; different training-set sizes mean different optimizer-update counts, so this is not a strict equal-compute causal comparison.

| Model | AP50–95 | AP50 | F2-selected precision | F2-selected recall | TP / FP / FN | Small recall |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Original P2 | 55.07% | 84.40% | 66.67% | 86.27% | 44 / 22 / 7 | 0 / 2 |
| Real-only fine-tune | 53.52% | 84.57% | 90.91% | 78.43% | 40 / 4 / 11 | 0 / 2 |
| Real + Blender | 53.59% | 82.72% | 72.88% | 84.31% | 43 / 16 / 8 | 0 / 2 |

The original `models/challenge-20261019/` remains the recommended default under the AP50–95 selection criterion. The synthetic-assisted model has only a 0.06 percentage-point AP lead over the matched control, trails the original by 1.49 points, and does not improve the two small targets. This is not evidence of a reliable synthetic-data benefit. The same validation set was reused for model and threshold selection; independent sites/encounters remain necessary.

For inspecting the completed Blender-trained candidate, `models/blender-experimental-20261019/` contains `turtle.pt`, `turtle.onnx`, and `turtle.mlpackage`, its full model card, data audit, environment versions, and comparison reports. Its selected threshold is `0.11974073201417923`; the card explicitly marks it as experimental and not the recommended default. ONNX and FP32 Core ML passed three-image raw-output checks: maximum coordinate differences were 0.00855 and 0.00593 pixels respectively, with score differences below 0.000003. Four-scale output shape is `[1, 5, 34000]`; host NMS is required. Full exported-model task metrics and physical-device inference remain unverified.

The candidate also replayed the second OAK recording locally: 76 frames, 71 with predictions, 83 proposed detections, 73 low-quality depth associations, and 10 unknown distances. Overlays and JSON are in `runs/challenge/blender_replay_205810/`. These are unreviewed predictions, not an accuracy measurement. The full comparison and figure are in `runs/challenge/comparison_blender_20261019/`.

```sh
.venv/bin/python predict_challenge.py path/to/image.jpg --model-card models/blender-experimental-20261019/model.json --output runs/challenge/blender-image-review
```

The individual workflow stages above were executed successfully for this experiment. The new one-command orchestration wrapper was checked for syntax and CLI arguments, but has not been rerun as a second complete duplicate training job. No OAK camera was opened, no original scan was edited, and no data or models were uploaded.

## Trained RGB-depth fusion, 2026-10-19

`models/fused-experimental-20261019/` now contains an actually trained two-input model: `turtle_fused.pt`, `turtle_fused.onnx`, model card, provenance, dataset/checksum manifests, learning curves, evaluation, and export parity evidence. Use `load_fused()` from `train_fusion.py` for this custom checkpoint; it is not an ordinary single-input `YOLO(...).predict()` checkpoint. The original RGB package remains the default pending real paired evaluation.

The run `runs/fusion/rgb_depth_p2_20261019/` trained 29,664 depth-encoder and fusion parameters for 20 fixed epochs, batch 4, on all 96 paired Blender scenes (84 positives, 12 backgrounds). The original P2 detector, including BatchNorm buffers, stayed frozen and was verified exactly unchanged. The depth encoder and each projection/gate received nonzero updates. The final epoch was selected in advance; no scan derivatives were moved to validation and no unreviewed OAK predictions became labels. Training completed locally on MPS in approximately 183 seconds.

The optimizer was AdamW, learning rate 0.001 with cosine decay to 0.0001, weight decay 0.0005, gradient clipping at 10, seed 42, full precision. MPS operations are not claimed bitwise reproducible. Each epoch balanced the simulated OAK/iPhone profiles, alternating the profile for every scene. Augmentation used shared image/geometry/box horizontal flips, image-only brightness/contrast, 20% complete depth dropout, 30% local-hole probability, and 20% alignment-perturbation probability (up to three pixels with no wrap). The profiles already contain simulated noise, limited resolution, missing measurements, and boundary errors. No multimodal mosaic or mixup was used.

The loss is the pinned Ultralytics 8.4.157 one-to-many detection loss with TAL top-k 10, STAL small-box handling, and YOLO26's L1 regression term; box/cls/dfl gains are 7.5/0.5/1.5. This stage intentionally omits the frozen, detached one-to-one branch's loss and progressive branch weighting: that branch cannot train the fusion encoder. The deployed inference path is explicitly one-to-many with host NMS. Installed source was inspected alongside the [official trainer customization documentation](https://docs.ultralytics.com/guides/custom-trainer).

Inputs are float32 `[B,3,640,640]` image and geometry tensors. Geometry is fixed log-normalized camera-axis depth over 0.2–10 m, validity, and quality. Invalid/padded data remain zero. Quality is explicitly binary validity, not calibrated confidence. Current StereoDepth documentation was successfully retrieved through Luxonis MCP and retained in `evidence/fusion-20261019/stereodepth-mcp.md`. The [RVC2 confidence convention](https://docs.luxonis.com/software-v3/depthai/depthai-components/nodes/stereo_depth) is inverted, and its raw confidence map is not assumed RGB-aligned; it is not passed into this model.

| Check / diagnostic | Result |
| --- | --- |
| 50 real validation photos with zero geometry | Exact raw-output equality with the frozen RGB detector; AP50–95 55.07% |
| 96 synthetic training scenes, zero depth | AP50–95 31.50% |
| Same training scenes, simulated OAK depth | AP50–95 62.32% |
| Same training scenes, simulated iPhone depth | AP50–95 58.62% |
| Same training scenes, OAK depth shifted 12 pixels | AP50–95 61.90% |
| Same training scenes, normalized depth reversed, validity preserved | AP50–95 61.88% |
| Geometry/fusion/training tests | 21 passed |
| ONNX CPU parity on three nonzero-depth fixtures plus one real zero-depth image | Passed: maximum box error 0.00524 px, score error below 0.000002 |

The synthetic scores are **training diagnostics, not validation or field accuracy**. Similar results after reversing depth mean these results do not isolate useful metric geometry reasoning; the quality gates and adaptation to synthetic appearance may explain much of the gain. The 20 cm shell scale is still assumed, sensor profiles are uncalibrated, and a single stylized incomplete shell is not species diversity. There are no labeled real paired training or validation samples, no paired iPhone captures, and no trained IR variant. The inherited RGB confidence threshold 0.07244315 has not been selected or calibrated for fused outputs.

The real OAK session `20261016T205810_f6a2ab4a` passed structural validation and was replayed offline through learned fusion: 76 frames, depth changed scores in all 76, 61 frames with predictions, and 74 proposed detections. Results are in `runs/fusion/rgb_depth_p2_20261019/oak_replay/`. Spot inspection of frame 50 shows both a turtle detection and a false shoe/leg detection. This is a functioning inference check, not a labeled accuracy assessment. Timestamps, alignment, validity, and original image coordinates are retained; range estimation remains separate from detection.

The ONNX graph has two inputs named `image` and `geometry`, each `[1,3,640,640]`, and raw `[1,5,34000]` xywh-plus-score output requiring host NMS. It passed ONNX checker and CPU numerical parity. It has not been compiled for RVC2 or deployed to a phone; no fused Core ML artifact or on-camera runtime claim is made. Use `.venv-coreml` for ONNX export dependencies and `.venv` for training/inference.

```sh
# Train another local experiment; choose a new run name.
.venv/bin/python train_fusion.py --name rgb_depth_next --epochs 20 --batch 4

# Replay saved, validated RGB-aligned OAK depth. No camera is opened.
.venv/bin/python predict_fusion.py recordings/oak/20261016T205810_f6a2ab4a --output runs/fusion/my-replay

# Single RGB image plus an already aligned camera-axis depth map in meters.
.venv/bin/python predict_fusion.py path/to/rgb.png --depth path/to/depth_m.npy --output runs/fusion/my-image

# Omitting --depth exercises the exact zero-geometry RGB fallback.
.venv/bin/python predict_fusion.py path/to/rgb.png --output runs/fusion/my-rgb-only

.venv/bin/python -m unittest discover -s tests/training -v
```

All files stayed local. Reviewed real RGB-depth labels, grouped by observation/individual and split across genuinely separate encounters, are the next requirement for real fine-tuning and a credible fusion-versus-RGB comparison.

## Gaussian-splat data, 2026-10-26

`datasets/turtle-gaussian-v1/` contains 96 new Gaussian-rendered views: 84 positive shell-proxy examples and 12 known backgrounds. The combined dataset references 198 real training photos, 96 existing Blender renders, and 96 new splat renders (390 training images total); its 50 real validation images and split list are unchanged. The fusion loader consumes the 192 synthetic RGB-depth pairs. Existing images are referenced rather than copied. This request generated data and checked model/loss compatibility; it did not retrain or replace the detector.

Generation uses [MetalSplat](https://github.com/tchauffi/metalsplat) 0.2.0 pinned to commit `84e8c3a433f71c107e9387d3f72a765436c11f6e`, installed into the existing training `.venv` without changing other dependencies. PyTorch 2.14.0 supplies its Metal backend. No CUDA service, external images, model downloads, or uploads are involved. COLMAP is not used, so its optional reconstruction dependency is unnecessary for this path.

For each of 24 selected labeled Blender scenes, `gaussian_dataset.py` backprojects every second valid source pixel into the source-camera coordinate frame and creates 102,400 anisotropic 3D Gaussians with color, opacity, scale, rotation, and shell identity. The standard Gaussian projection and front-to-back alpha compositing produce four nearby views per source. See the [gsplat rasterization specification](https://docs.gsplat.studio/main/apis/rasterization.html) for the projection/compositing convention. Camera yaw/pitch stay within 0.65 degrees and translations are 0.1–0.2% of the scene's median Z. World-to-camera transforms are composed and saved, with the source intrinsics preserved and pixel centers at `(u+0.5,v+0.5)`.

This is **RGB-D initialized Gaussian-splat augmentation**, not a multiview-optimized reconstruction: Gaussian positions, colors, and opacity are not fitted from photographs. The input contains baked shading, and the method cannot recover unseen shell sides. The small camera motions limit disocclusion. It adds nearby views of the existing scenes, not new animals, materials, independent observations, or proven photorealism. The scan scale remains an assumed 20 cm. All derivatives retain the original scan group and training partition.

The same visibility compositing produces RGB, shell contribution, depth first moments, and depth second moments. The pinned Metal kernel returns an accumulated depth sum; the adapter divides by accumulated alpha exactly once to obtain expected camera-axis Z. A known-plane fixture verifies this, including partial alpha. This expectation is still rendered depth, not a measured physical surface. Pixels with alpha below 0.95 or depth standard deviation above `max(0.005 m, 0.02 × Z)` are marked unknown. At least 95% RGB coverage is required for accepting a view; the completed batch's minimum was 97.93%. Mean valid rendered depth was 96.62% before sensor corruption.

Visible boxes enclose pixels where the alpha-weighted shell contribution is at least half of total support. Foreground occluders participate in the same depth ordering as RGB. Labels are independent of depth validity: 22,676 shell pixels across this dataset retain their RGB labels while their mixed-surface depth is unknown. A positive source that becomes too small/invisible is rejected rather than converted into a background. Existing simulated OAK/iPhone degradation is then applied, preserving all unknown pixels. These profiles remain explicitly uncalibrated simulations; the average valid support after corruption was 70.93%.

| Artifact | Purpose |
| --- | --- |
| `datasets/turtle-gaussian-v1/manifest.json`, `data.yaml` | Combined training inputs for fusion and RGB experiments |
| `datasets/turtle-gaussian-v1/contact-sheet.jpg` | First view of each of the 24 source scenes, with boxes |
| `datasets/turtle-gaussian-v1/splats/*.npz` | Reusable Gaussian means/scales/quaternions/opacities/colors and shell identity in each source-camera frame |
| `datasets/turtle-gaussian-v1/splats/example.ply` | One standard 3DGS PLY example for external viewers; semantic IDs remain in NPZ |
| `datasets/turtle-gaussian-v1/samples/*/sample.json` | Camera transforms, source hashes, label/depth semantics and quality |
| `datasets/turtle-gaussian-v1/samples/*/depth.npy` | Rendered expected camera-axis meters, zero where unknown |
| `datasets/turtle-gaussian-v1/samples/*/{oak,iphone}_depth_z_m.npy` | Simulated sensor profiles consumed by the fusion loader |
| `datasets/turtle-gaussian-v1/validation.json`, `artifact_checksums.json` | Actual dataset checks and file hashes |
| `evidence/gaussian-20261026/` | Test, generation, validation, and visual QA evidence |

All 96 requested views were accepted; none were rejected. The batch occupies about 727 MiB locally. Twenty-seven tests passed, including six new splat geometry/raster tests for pixel centers, unknown depth, pose validity, sampling, alpha normalization, occlusion, and labels under invalid depth. Both sensor profiles for all 96 new views passed the actual fusion loader. Validation also checks source hashes, exact visible-box/YOLO correspondence, camera composition, unknown-depth preservation, and the unchanged real validation split. A two-sample batch passed the trained fusion model's forward and training-loss paths, with no optimizer steps or saved model changes. This establishes pipeline compatibility, not an accuracy improvement.

```sh
# Reproduce locally; the renderer refuses an existing output directory.
uv pip install --python .venv/bin/python --no-deps -r requirements-gaussian.txt
.venv/bin/python gaussian_dataset.py --output datasets/turtle-gaussian-next --source-count 24 --views 4
.venv/bin/python validate_gaussian_dataset.py datasets/turtle-gaussian-next --check-model models/fused-experimental-20261019/turtle_fused.pt

# Use the completed combined dataset for a future fusion training experiment.
# This initializes from the original RGB P2 weights and trains the fusion branch.
.venv/bin/python train_fusion.py --name rgb_depth_gaussian_next --manifest datasets/turtle-gaussian-v1/manifest.json --epochs 20 --batch 4

# Alternatively compare an RGB model using exactly the same combined split.
.venv/bin/python train_challenge.py --variant p2 --name p2_gaussian_next --data datasets/turtle-gaussian-v1/data.yaml --init-checkpoint models/challenge-20261019/turtle.pt --epochs 20 --learning-rate 0.00005
```

The renderer checks disk space and retains a 1.5 GB free-space reserve by default. Generation uses the labeled synthetic sources because the real OAK recordings do not yet have reviewed masks/boxes or estimated multiview poses suitable for semantic reconstruction. Their model predictions were not repurposed as ground truth. Genuine real-scene reconstruction would require suitable overlapping static captures, camera poses, and reviewed annotation transfer before it could supply labeled training data.

## Gaussian-augmented fusion training, 2026-10-26

`runs/fusion/rgb_depth_gaussian_20261026/` completed 20 fixed epochs on 192 paired samples: 96 original Blender renders plus 96 Gaussian-splat derivatives (168 positive images, 24 known backgrounds). The 198 real photos contributed through the pretrained RGB detector, not invented depth pairs. The 29,664 depth/fusion parameters were freshly initialized from the original RGB P2 checkpoint, using the same frozen-RGB recipe as the first fusion experiment: batch 4, AdamW 0.001, cosine decay, seed 42, depth dropout/holes/alignment perturbations, and both simulated sensor profiles. All RGB parameters and BatchNorm buffers were verified exactly unchanged after training. The final epoch was selected in advance; no synthetic frame was moved into validation.

Training completed in 204 seconds on local MPS. The first/final augmented losses were 4.998 and 3.279. Both the previous and new models were then evaluated on identical inputs using CPU inference and host NMS. The table reports AP50–95; synthetic rows are **training-scene diagnostics**, not independent testing.

| Identical evaluation inputs | Previous fusion | Gaussian-augmented fusion |
| --- | --- | --- |
| 50 real photos, zero depth | 55.07% | 55.07% |
| 192 synthetic examples, zero depth | 35.69% | 35.69% |
| 192 synthetic examples, OAK-like depth | 66.50% | 73.08% |
| 192 synthetic examples, iPhone-like depth | 63.33% | 70.01% |
| 96 original Blender examples, OAK-like depth | 62.32% | 64.61% |
| 96 Gaussian views, OAK-like depth | 71.10% | 81.66% |

The real zero-depth outputs matched the frozen RGB detector exactly on all 50 images. The synthetic improvement establishes better fit to these inputs, not a field improvement or an isolated benefit of Gaussian augmentation: the new run has twice as many optimizer updates, the Gaussian views are correlated derivatives, and the same single scan underlies all pairs. Reversing normalized depth while preserving validity still yields 73.04% combined AP, close to ordinary depth's 73.08%; useful metric-geometry reasoning remains unproven. Independent reviewed RGB-depth encounters are still needed. The inherited RGB threshold remains 0.07244315 and is not calibrated for the new fused scores.

`models/fused-gaussian-20261026/` contains `turtle_fused.pt`, `turtle_fused.onnx`, the model card, provenance, manifests/checksums, learning curves, same-input previous-model comparison, and export report. ONNX parity passed seven fixtures spanning Blender, Gaussian-splat, OAK-like/iPhone-like depth, and a real zero-depth image. Maximum raw coordinate error was 0.00779 pixels and score error below 0.0000023. The graph retains two `[1,3,640,640]` inputs and `[1,5,34000]` raw output requiring host NMS. The training suite passed all 27 tests. No fused Core ML or on-camera compilation is claimed.

Commands for this completed experiment and its saved-session replay:

```sh
# Already executed; choose a fresh run name if repeating.
.venv/bin/python train_fusion.py --name rgb_depth_gaussian_20261026 --manifest datasets/turtle-gaussian-v1/manifest.json --epochs 20 --batch 4

# Use the new model explicitly; choose a fresh output directory.
.venv/bin/python predict_fusion.py recordings/oak/20261016T205810_f6a2ab4a --model-card models/fused-gaussian-20261026/model.json --output runs/fusion/gaussian-replay-next
```

The actual offline replay is under `runs/fusion/rgb_depth_gaussian_20261026/oak_replay/`; these predictions are not reviewed labels or a real-depth accuracy test. The previous model remains available, the original RGB default is unchanged, and all data and computation stayed local. Experiment evidence is under `evidence/fusion-gaussian-20261026/`.

## Scan and splat training, 40 epochs, 2026-10-26

`models/fused-scan-splat-40ep-20261026/` contains a new completed experimental checkpoint and verified two-input ONNX export. The source `assets/Scan.obj` still matches its recorded SHA-256. Dataset validation passed before training: 96 Blender scan renders plus 96 RGB-D-initialized Gaussian-splat views, with 168 positive images and 24 backgrounds. All scan derivatives remain training-only; the real validation split is unchanged. The Gaussian representation was rasterized for augmentation, not optimized as a multiview reconstruction during this run.

The run `runs/fusion/rgb_depth_scan_splat_40ep_20261026/` trained the 29,664 depth/fusion parameters for 40 fixed epochs, batch 4, seed 42, AdamW at 0.001 with cosine decay to 0.0001. It started afresh from the original RGB P2 detector, not from the previous fused checkpoint. The data, augmentation and initialization match the 20-epoch experiment, but the optimizer-update count doubles and the cosine schedule spans 40 epochs. All frozen RGB parameters and BatchNorm buffers were verified exactly unchanged. Training took 395 seconds locally on MPS; first/final augmented loss was 4.998 / 2.697. The final epoch was specified before training and was not selected using these diagnostic scores.

| Same evaluation inputs; AP50–95 | Previous 20 epochs | New 40 epochs |
| --- | ---: | ---: |
| 50 real photos, zero depth | 55.07% | 55.07% |
| 192 synthetic training scenes, OAK-like depth | 73.08% | 77.21% |
| 192 synthetic training scenes, iPhone-like depth | 70.01% | 74.60% |
| 96 Blender scenes, OAK-like depth | 64.61% | 70.63% |
| 96 Gaussian views, OAK-like depth | 81.66% | 84.05% |
| 192 synthetic scenes, OAK depth shifted 12 px | 72.14% | 75.62% |
| 192 synthetic scenes, normalized depth reversed | 73.04% | 76.85% |

These are better fits to training scenes, not independent validation or evidence that Gaussian augmentation itself caused an improvement. Reversed depth still performs similarly to ordinary depth, so useful metric-depth reasoning remains unproven. The single incomplete scan, assumed 20 cm scale, uncalibrated sensor simulation, inherited uncalibrated detection threshold, and lack of reviewed real paired encounters remain limitations. No new recommended default or deployment claim is made.

All 27 training tests passed. Real zero-depth raw outputs matched the frozen RGB detector exactly on all 50 photos. ONNX checker and CPU parity passed seven fixtures spanning both synthetic domains and real zero-depth input: maximum coordinate error 0.00809 px and score error 0.00000233. The graph has two `[1,3,640,640]` inputs and raw `[1,5,34000]` output requiring host NMS; it is not compiled for OAK or deployed to a phone. No new camera session or saved-session replay was needed for this training request. Evidence and checksums are under `evidence/fusion-scan-splat-40ep-20261026/`.

```sh
# Executed training command; use a fresh name to repeat.
.venv/bin/python train_fusion.py --name rgb_depth_scan_splat_40ep_20261026 --manifest datasets/turtle-gaussian-v1/manifest.json --epochs 40 --batch 4

# Use the new trained model explicitly on an aligned RGB-depth pair.
.venv/bin/python predict_fusion.py path/to/rgb.png --depth path/to/depth_m.npy --model-card models/fused-scan-splat-40ep-20261026/model.json --output runs/fusion/scan-splat-image-review
```
