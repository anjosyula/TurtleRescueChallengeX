> Historical local research notes. For the portable OAK demo and bundled RT-DETR models, use [the current README](../README.md). Historical output/dataset paths below require separately supplied assets.

# TurtleHero

Read [instructions.md](../instructions.md) for the proposed multimodal detector architecture.

- [OAK-D Pro collection guide](../OAK_CAPTURE.md): collect synchronized RGB, mono, and depth data from live scenes; review turtle boxes locally.
- [Training and iPhone guide](../TRAINING.md#challenge-experiments-2026-10-19): audited RGB/P2 experiments, evaluation, exports, and remaining data requirements. The earlier iOS integration is documented separately in that guide.
- [GMU Hopper deployment](../cluster/hopper/README.md): prepare a verified transfer list, build the Linux CUDA environment, submit RGB or fusion Slurm jobs, and resume interrupted runs. No account or data transfer is needed to prepare locally.

Use `train_challenge.py` for new training experiments. It excludes the confirmed cross-split near-duplicate and records the data, configuration, and checkpoint provenance:

```sh
.venv/bin/python train_challenge.py --variant rgb --name rgb_repeat
.venv/bin/python train_challenge.py --variant p2 --name p2_repeat
```

Use the selected local checkpoint with `predict_challenge.py`:

```sh
.venv/bin/python predict_challenge.py path/to/image.jpg --output runs/challenge/my-image
```

The default RGB model is in `models/challenge-20261019/`; its model card records the selected confidence threshold and validation limitations. A separately trained RGB-depth experiment is in `models/fused-experimental-20261019/`, including its two-input PyTorch checkpoint and verified ONNX export. It trained the fusion branch on 96 simulated pairs while freezing the RGB detector. Real paired fine-tuning and validation remain outstanding. See [fusion training and replay](../TRAINING.md#trained-rgb-depth-fusion-2026-10-19). Predictions remain review candidates, not ground-truth annotations.

The small-scale fine-tuning follow-up did not improve AP or small-target recall, so the original packaged model remains selected. Two [OAK review queues](../OAK_CAPTURE.md#review-with-detector-proposals) are ready, with 48 frames selected for correcting detections and missed turtles before paired-data training.

The [Blender workflow](../TRAINING.md#blender-synthetic-data-workflow-2026-10-19) includes an editable shell scene, RGB/depth/mask rendering, dataset validation, a real-only training control, a synthetic-assisted experiment, evaluation, and export. Reproduce it with `.venv/bin/python run_blender_workflow.py --tag blender_next`. The default 20 cm shell length is an explicit simulation assumption.

The [Gaussian-splat data workflow](../TRAINING.md#gaussian-splat-data-2026-10-26) generated 96 additional labeled RGB–depth views in `datasets/turtle-gaussian-v1/`. It uses genuine 3D Gaussian rasterization initialized from the existing labeled Blender RGB-D scenes, with small camera offsets. Its combined manifest works with the fusion trainer and preserves the real validation split. These are correlated synthetic augmentations, not a fitted real-scene reconstruction or additional independent turtles.

The subsequent [Gaussian-augmented fusion experiment](../TRAINING.md#gaussian-augmented-fusion-training-2026-10-26) trained for 20 epochs on all 192 paired synthetic examples. Its PyTorch checkpoint, verified two-input ONNX export, model card, and comparisons are in `models/fused-gaussian-20261026/`. Supply `--model-card models/fused-gaussian-20261026/model.json` to `predict_fusion.py` to use this experiment. Real paired accuracy remains unmeasured.

A further [40-epoch scan-and-splat experiment](../TRAINING.md#scan-and-splat-training-40-epochs-2026-10-26) is saved in `models/fused-scan-splat-40ep-20261026/`, including PyTorch, verified ONNX, and comparison reports. It uses the same 96 scan renders and 96 splat derivatives. Supply `--model-card models/fused-scan-splat-40ep-20261026/model.json` to select it. Its higher synthetic training scores do not establish field improvement; the RGB default remains unchanged.

The OAK recorder is device-tested acquisition code. It does not run a detector, train the proposed depth-fusion network, or establish field safety performance.

## Photo-fitted shell templates

The new `What About The Turtles Photos` archive is preserved and indexed under `datasets/what-about-turtles/`: 133 still images, 9 videos and one document. `media_manifest.json` stores original hashes, capture metadata and review status; `images-1.jpg` through `images-5.jpg` and `videos.jpg` provide contact sheets. Metadata identifies 50 iPhone 11 and 46 iPhone 15 Pro media files, plus other models and 24 unknowns. There is no universal iPhone calibration. Originals remain unmodified, and no unreviewed image has been added to detector training. Import is reproducible with `ingest_turtle_archive.py` and `requirements-ingest.txt`; FFmpeg/ffprobe are also required. Two short construction-site sequences are prepared for the [experimental Hopper reconstruction job](../cluster/hopper/README.md#construction-site-gaussian-reconstruction-candidates).

The supplied [self-righting paper](https://www.researchgate.net/publication/5905508_Geometry_and_self-righting_of_turtles) provides a useful morphology prior: a transverse model with height/width ratio, carapace curvature, and rim rounding, fitted to 30 individuals across 17 species. These parameters can constrain future correlated shape variation; they are not interchangeable with the current mesh taper or its unverified plastron height. The [Natural History explanation](https://www.naturalhistorymag.com/biomechanics/10309/the-living-g-mb-c) provides context. Neither source supplies shell textures or justifies making every turtle a Gömböc. Definitions, regression references and limitations are recorded in `docs/shell-morphology-sources.json`; existing fitted renders and training are unchanged.

`output/shell-fit-20261026/comparison.png` shows the source photographs, fitted scan, randomized proportions, and actual Gaussian rasterization. Two source images produced 12 Blender variants and 12 Gaussian variants. These are review examples, not additions to the training manifest. The scan is a deformable template rather than a measurement of either photographed animal. Photo color is projected through the fitted camera onto source-visible mesh faces; unsupported faces remain gray. Captured illumination is retained, so novel lighting is not simulated by these examples.

The OAK example uses saved undistorted RGB intrinsics, binary depth validity, timestamps, and a robust interior stereo-depth summary. Its estimated length is 9.8 cm and convex-outline RMS error is 1.49 px; neither is a verified anatomical measurement. The downloaded snapping-turtle photo uses three uncalibrated focal hypotheses, an assumed 28 cm length, and an approximate traced outline (14.46 px contour RMS). The whole shell is mapped, replacing the earlier arbitrary inner-photo-patch mapping. Single-view scale, shape, focal length, and pose remain ambiguous.

Length priors refer to [Massachusetts' eastern box turtle profile](https://www.mass.gov/info-details/eastern-box-turtle), [NC Wildlife's hatchling/adult description](https://www.ncwildlife.gov/eastern-box-turtle-species-profile/open), and [Texas Turtles' common snapping turtle profile](https://www.texasturtles.org/common-snapping-turtle). Width/height ratios and taper limits are explicitly engineering augmentation bounds, not published species measurements. The source photo retains Lee Stayton's CC BY attribution from the download manifest. All derivatives retain source-group provenance; source masks are agent-traced estimates, not reviewed detector ground truth.

The code is `shell_fit.py`, `render_fitted_shells.py`, `validate_fitted_shells_blender.py`, and `fitted_shell_splats.py`. The additional fitting dependency is pinned in `requirements-shell-fit.txt`. Geometry and fusion tests passed 32 checks. All 12 actual Blender renders passed camera/depth round trips: maximum projection error 0.025 px and depth error below 0.000001 m. Visible photo-supported shell coverage was 93.3–98.0%. These numerical checks do not validate realism. Original scans, recordings, and trained models are unchanged.

```sh
# The exported template and configuration are retained with the completed example.
# Choose fresh output folders when repeating these commands.
.venv/bin/python shell_fit.py --template output/shell-fit-20261026/template.npz --config output/shell-fit-20261026/config.json --output output/shell-fit-next/fitted
/Applications/Blender.app/Contents/MacOS/Blender --background --python render_fitted_shells.py -- --fits output/shell-fit-next/fitted/fits.json --output output/shell-fit-next/variants
/Applications/Blender.app/Contents/MacOS/Blender --background --python validate_fitted_shells_blender.py -- output/shell-fit-next/variants
.venv/bin/python fitted_shell_splats.py output/shell-fit-next/variants --output output/shell-fit-next/gaussian
```

## Construction-site walk for reconstruction

The user-provided `splatposter.JPG` describes a real-video-to-splat environment for humanoid locomotion; it supports the idea of reconstructing realistic site backgrounds. `splattingoctopus.JPG` describes calibrated multiview masks, a visual hull, multiview stereo, and 2D Gaussian splatting for a moving octopus. Its segmentation and surface-reconstruction stages are relevant to turtle assets; underwater refraction, ultrasound, and locomotion rewards are not required for this detector. Exact public implementations of these two poster projects were not located in the searches performed. The poster results are not evidence of turtle-detection accuracy.

For the upcoming walk, start with a short pilot of one representative patch and inspect sharpness/overlap before covering the site. Walk a loop with translation and overlapping views, then add a lower, ground-facing pass; include the same rocks, ground patches, and vegetation edges from several angles. Keep one lens and avoid zoom/lens switches, cinematic effects, or changing crop modes. Preserve original files, camera metadata, and any pose/IMU/depth export. Record a known physical length for scale. Prefer stable illumination and scene elements; moving workers, equipment and windblown foliage need masking or exclusion during reconstruction. These are capture recommendations, not a guarantee that a casual walk will reconstruct successfully. [COLMAP capture guidance](https://colmap.github.io/tutorial.html) recommends high overlap, multiple views, good texture, and similar illumination.

An ordinary phone video can feed COLMAP camera-pose estimation and then a Gaussian optimizer; a mapping-aware recording can retain visual/inertial data for pose recovery. [Nerfstudio's supported data formats](https://docs.nerf.studio/quickstart/custom_dataset.html) include video, images, Record3D, Polycam and Spectacular AI. Device/app choice remains pending. The current `collect_oak.py` recordings contain synchronized RGB, mono and depth, but no IMU stream or camera trajectory, so they are not a complete visual-inertial mapping recording. A dedicated OAK mapping recorder would need separate setup and a short validated pilot; never run it concurrently with the current recorder. [Spectacular AI](https://www.spectacularai.com/mapping) documents iPhone/OAK recording and local processing, but its tools require commercial licensing for commercial use; no installation or license purchase was performed.

The intended next pipeline is: recover and verify poses/scale; fit a static site splat; establish geometry for consistent placement and occlusion; insert photo-fitted shell variants; render paired RGB, synthetic camera-axis depth, visibility labels and invalid-depth masks; apply explicitly simulated sensor corruption; compare with held-out real encounters. Reserve separate sites/encounters for testing. Do not place one source capture's derived views in both training and validation. 2DGS can be evaluated for geometry quality when multiview data exists; it is not the current MetalSplat 3D renderer. [2DGS reference implementation](https://github.com/hbb1/2d-gaussian-splatting), [SAM 3 segmentation](https://github.com/facebookresearch/sam3), and [Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html) are candidate components, not installed or tested integrations here.

The poster's full Isaac Sim/Isaac Lab workflow is unnecessary for this data generator. Its NVIDIA hardware requirements are different from the current Mac's Metal renderer. The first local site reconstruction experiment is now complete below; no data was uploaded and no turtle detector was retrained. Check free space before importing a substantial walk.

## First real site splat and OAK depth snapshot

`output/site-splat-20261026/` contains the actual local experiments. Arrowbrook recovered 40/40 cameras and 3,601 sparse points using CPU PyCOLMAP 4.2.1. Mean reprojection error was 0.339 px and median maximum track triangulation angle 7.77 degrees. A 2,500-step MetalSplat fit completed; the selected 6,848-Gaussian model is `arrowbrook-splat/garden_best.ply`. See `arrowbrook-splat/preview.jpg`, `comparison.jpg`, and `preview.mp4` for recorded RGB versus the fitted scene. This is a recognizable but rough reconstruction with blurred vegetation, stretched ground details and possible moving-equipment artifacts. It is not yet accepted for detector training or geometric placement. Scale remains unknown.

Five same-clip frames were withheld from photometric optimization, but all frames participated in SfM and point initialization. Their mean PSNR after PLY reload is 21.43 dB, a correlated diagnostic rather than independent novel-view validation. The best checkpoint occurred at step 750; the final checkpoint is also retained. Hartland produced partial components of up to 21/36 frames and failed the 80% registration gate, so no Hartland splat was fitted.

Both original OAK sessions passed structural revalidation. Session `20261016T205810_f6a2ab4a` has 54.67% mean valid depth; the earlier session has only 0.74%. `oak_snapshot_splat.py` converted sample `000050` of the better session into 192,480 Gaussians from actual RGB-aligned camera-axis depth and recorded intrinsics. See `oak-snapshot/comparison.jpg` and `oak-snapshot/oak_snapshot.ply`. Unknown source depth was not completed, raw confidence was not repurposed, and source Z coordinates round-trip exactly. This is a single-view metric snapshot, not a multi-view optimized turtle model. Rendered new-view depth is an estimate from the splats, not another sensor measurement. The moving turtle requires segmentation and object-motion handling before observations from multiple timestamps can be combined.

Reproduce into fresh output directories:

```sh
uv pip install --python .venv/bin/python --no-deps -r requirements-site-splat.txt
.venv/bin/python local_site_sfm.py --scene datasets/site-candidates/arrowbrook-20211021 --output output/site-next/arrowbrook
.venv/bin/python fit_site_splat.py --reconstruction output/site-next/arrowbrook --output output/site-next/fit --steps 2500
.venv/bin/python preview_site_splat.py --reconstruction output/site-next/arrowbrook --fit output/site-next/fit
.venv/bin/python oak_snapshot_splat.py --session recordings/oak/20261016T205810_f6a2ab4a --sample 000050 --output output/site-next/oak-snapshot
```

The fit wrapper uses the pinned upstream training example and MIT license under `vendor/metalsplat_site/`. Source hashes, settings, frame provenance, logs, original camera reconstruction and verification reports are retained with outputs. The site optimizer's residual-based background initialization hypothesizes unobserved depth; its geometry must never be labeled measured OAK data. The OAK snapshot adapter does not use that completion mechanism.

## Additional model candidates

The separate RGB/depth OR and RT-DETR pilot is reproducible with `train_rtdetr_pilot.py`, `train_depth_pilot.py`, and `evaluate_dual_pilot.py`. It uses the existing `.venv` and pinned Ultralytics implementation, not the NVIDIA TAO runtime. The RGB challenger uses the existing 198/50 image split with square letterboxes; the depth-only detector receives normalized depth and validity, never RGB, and trains only on existing simulations. All derivatives of the scan remain training-only. Saved OAK frames remain unreviewed; their replay alert counts cannot measure accuracy or false alerts per minute. No default detector is replaced automatically.

The completed 2026-10-26 experiment is `output/dual-detector-20261026/summary.json`. Both new detectors trained for 10 epochs: RT-DETR-L with frozen backbone weights on 198 real RGB photos, and an independent YOLO26n-P2 geometry-only detector on 192 simulated scenes. At 640-square input, the common 50-image/51-turtle validation measured AP50–95 of 0.6722 for RT-DETR versus 0.5512 for the current RGB YOLO. Their validation-selected thresholds each found 44/51 turtles, with 9 versus 22 false boxes. Median host tensor-to-decoded-output latency was 113.3 versus 46.6 ms, excluding acquisition and preprocessing. This is a short checkpoint comparison with different training budgets, not independent field validation or an OAK/phone runtime benchmark.

All 159 saved OAK frames were replayed. RGB and OR each alerted on 154 frames; the depth-only model alerted on 29, adding zero new alert frames. Missing-depth decisions preserved RGB alerts on every frame. The recordings remain unreviewed, so these counts do not measure correctness. On the simulated OAK training scenes, boolean OR alerts covered 165/168 positive frames versus RGB's 155/168, with 13/24 negative frames alerting in both cases. The box-level OR result includes duplicate detections and must not be confused with frame-level false alerts. Depth-value reversal reduced the geometry model's diagnostic AP from 0.7368 to 0.6474; real generalization remains unproven. Current defaults are unchanged; 44 training tests and checkpoint forward/backward/reload checks passed.

See `evaluation/rgb_comparison.json`, `evaluation/synthetic_diagnostics.json`, and `evaluation/oak_replay_summary.json` within that experiment. Each OAK subfolder contains prediction records and a `contact.jpg`; the report distinguishes branch scores and unavailable depth. `report_dual_pilot.py` regenerates the summary from saved predictions. The earlier measured-depth splat is still available at `output/site-splat-20261026/oak-snapshot/comparison.jpg` and `oak_snapshot.ply`.

```sh
# Use a fresh experiment folder. The official rtdetr-l.pt checkpoint must be local.
.venv/bin/python train_rtdetr_pilot.py --output output/dual-pilot-next/rtdetr --epochs 10
.venv/bin/python train_depth_pilot.py --output output/dual-pilot-next/depth --epochs 10
.venv/bin/python evaluate_dual_pilot.py --experiment output/dual-pilot-next
.venv/bin/python report_dual_pilot.py output/dual-pilot-next
```

`turtle_late_fusion.py` implements the OR alert policy with separate thresholds, branch provenance, cross-branch duplicate association and explicit unavailable-depth status. Depth cannot veto RGB. The experimental depth threshold of 0.25 is not field-calibrated. RT-DETR's short frozen-backbone pilot and the existing YOLO baseline have different training budgets, so the comparison is between checkpoints, not a definitive architecture ranking.

Luxonis MCP documentation confirms that calibrated stereo supplies metric depth and the current PointCloud node can output meters. Our OAK snapshot already uses saved metric camera-axis Z and per-frame intrinsics. RGB intrinsics alone do not determine absolute scale. Scaling a separate RGB reconstruction requires shared static observations with measured depth or a known physical reference; the older phone-only Arrowbrook clip lacks such an anchor. See `docs/oak-metric-scale.json` and the [PointCloud documentation](https://docs.luxonis.com/software-v3/depthai/depthai-components/host_nodes/pointcloud). Metric units do not establish shell-boundary accuracy or recover hidden anatomy.

[CAL](https://research.nvidia.com/labs/dvl/projects/complete-anything-lidar/) contributes useful ideas for tracking video masks and aggregating calibrated partial 3D observations. Its deployed model consumes LiDAR and infers missing geometry; effectiveness on small turtles and stereo-derived points is unproven. The [NGC Transfusion card](https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/transfusion/-) actually describes a BEVFusion-based people model, not a turtle model or a splat renderer. See `docs/nvidia-3d-references.json` for the assessment; neither model was downloaded or installed.

The user's proposed separate RGB/depth detectors with an OR alert is recorded as an experiment, not an implemented/trained capability. It can increase recall at fixed branch thresholds while admitting false positives from either model; missing depth must remain an unavailable branch. Evaluate it against the existing fused detector at a matched false-alert budget. [TAO RT-DETR](https://docs.nvidia.com/tao/tao-toolkit/latest/text/cv_finetuning/pytorch/object_detection/rt_detr.html) is an RGB detector candidate with COCO-format annotations and configurable backbones. It is not established as better than this project's YOLO26-P2 on turtles. The [original RT-DETR paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.pdf) explicitly notes remaining small-object limitations. Compare small/occluded-target recall, false alerts per minute, and target-device latency on shared independent encounters before replacing the baseline.


## Live camera dashboard and peer demo

Connect an OAK camera and run `./start_dashboard.command`. Open `http://localhost:8765` on the laptop, then open the printed LAN address on another computer on the same network. The dashboard shows annotated RT-DETR detections, source station, recent sightings, camera status and direct peer connections. A browser-only observer needs no installation.

Use `./start_dashboard.command --source replay` for visibly labeled saved OAK footage. For a separate relay server, run `python3 turtlehero_dashboard.py --source none --peer http://CAMERA_IP:8765`. See [the dashboard guide](../docs/dashboard.md) for launch, troubleshooting, peer setup, limits and verification. The real detector and two local server instances were exercised; live OAK and a second physical computer remain to be verified with connected hardware.

## USB background update (2026-11-06)

The user-confirmed construction-site recording now supplies 29 background training frames for a completed five-epoch RT-DETR fine-tune. Use `start_dashboard_usb.command` for the new model or `replay_usb_background.command` to inspect the recorded background. The original checkpoint remains the normal dashboard default because the new model did not pass both comparison criteria. The USB launcher explicitly selects the new checkpoint.

See [the training comparison and measured fence splat](../docs/usb-background.md) for results, provenance and reproduction commands.
