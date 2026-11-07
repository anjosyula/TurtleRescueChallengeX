> Historical local experiment notes. Raw recordings and output folders described below are not included in this GitHub checkout. Use the root README and `run_dashboard.py --model usb` for the portable camera demo.

# USB background training, November 6, 2026

The user confirmed the October 31 OAK recording contains no turtles. We validated all 117 RGB/depth samples and copied RGB frames plus provenance into `datasets/usb-background-20261106`. Original USB files remain unchanged. The earlier failed USB boot has no samples and was excluded.

Twenty-nine spaced background frames joined the existing 198 labeled training photos. Empty labels reflect the user's explicit confirmation. All views of this USB encounter stay in training; the existing 50-photo validation set is unchanged and has no file-hash overlap. These 50 photos are the existing model-selection set, not an independent field evaluation.

RT-DETR-L was fine-tuned for five epochs on this laptop's CPU, starting from the existing turtle checkpoint. The first 10 layers were frozen, with AdamW at learning rate 0.00002. This is the project's Ultralytics RT-DETR-L implementation, not a newly executed NVIDIA TAO training job. No data was uploaded.

## Measured comparison

Both checkpoints used the same preprocessing and decoder on the same 50 photos (51 annotated turtles).

| Measurement | Original | USB-adapted |
|---|---:|---:|
| AP50–95 | 0.6722 | 0.6766 |
| Turtles detected at up to nine false boxes | 44/51 | 43/51 |
| Actual false boxes at that operating point | 9 | 6 |
| USB frames flagged at the unchanged score threshold 0.633166 | 42/117 | 8/117 |

USB counts are diagnostics on the training encounter. They do not establish unseen-site performance or false alerts per hour. The default-selection check requires AP50–95 and true positives at the nine-false-box budget both to equal or exceed the original result.

The original checkpoint remains the normal dashboard default because the new model did not pass both comparison criteria. The USB launcher explicitly selects the new checkpoint. The new launch threshold is 0.576352417, selected on the validation set. The UI can adjust it. See `output/usb-background-20261106/comparison.json` for fixed-threshold counts and all operating points.

## Use in the existing dashboard

Double-click `start_dashboard_usb.command` for the new model and a live OAK camera. The usual LAN address and peer connections still work. For the confirmed background recording, double-click `replay_usb_background.command`. This runs real inference over the copied USB images and labels the feed as recorded footage; it is not a live camera or a turtle-positive demonstration.

The main USB launcher retains the existing positive OAK replay as its normal Saved OAK footage option. The dedicated background launcher selects the new recording instead. Stop one dashboard before starting another on the same port. `docs/dashboard.md` describes the two-computer setup.

A newly attached OAK camera was not found during discovery. No new live recording was made. `record_oak_usb.command` remains available once the camera is connected; stop the dashboard before recording.

## Artifacts and reproduction

- Packaged checkpoint: `models/rtdetr-usb-background-20261106/turtle.pt` and `model.json`.
- Training settings, input hashes, exact executed training source, progress and results: `output/usb-background-20261106/`.
- Timestamp-paced RGB replay: `output/usb-background-20261106/rgb-replay.mp4` (57.72 seconds of saved RGB observations).
- All 117 local RGB frames and source metadata: `datasets/usb-background-20261106/`.
- CPU Gaussian generation/rendering: `render_oak_splat_cpu.py`.

To repeat training, choose fresh dataset and output directories:

```sh
.venv/bin/python train_usb_background.py --phase all --device cpu --epochs 5 \
  --dataset datasets/usb-background-repeat --output output/usb-background-repeat --model-output models/usb-background-repeat
```

The evaluator refuses to overwrite an existing model package; choose a fresh `--model-output` destination for each repeat.

## Construction-fence Gaussian snapshot

`output/usb-fence-splat-20261106/` contains a PLY with 76,164 Gaussians, a four-second moving view and measured source/projection checks. Frame 000085 supplies aligned OAK camera-axis depth in meters. Accepted depth is 0.2–3 meters, sampled at a stride of two pixels. The source frame shows a construction perimeter fence with fabric screen; it is not verified to be a silt fence.

This is a single-frame measured-depth snapshot, not an optimized multiview reconstruction or a complete map. Black gaps have no accepted measured depth. The virtual camera moves at most 2.5 cm. No rendered novel views were used for the detector fine-tune. A fresh CPU rebuild reproduced the PLY, stills and video byte for byte.

```sh
.venv/bin/python render_oak_splat_cpu.py \
  --session /Volumes/ESD-USB/recordings/oak/20261031T151948_a6eff4b9 \
  --sample 000085 --output output/usb-fence-repeat
```

The presentation includes this snapshot and preserves the earlier phone-video silt-fence reconstructions. Team names remain for the user to provide.

## Dashboard verification

The packaged checkpoint produced a real turtle detection from the existing OAK replay and passed the image, source timestamp and station identity to a second local HTTP dashboard. A separate pass delivered the copied USB background frame with zero detections on that observed frame. Both feeds stayed labeled recorded footage. All 13 dashboard regression tests passed. This verifies local peer transfer, not a second physical computer or live USB camera. Evidence: `output/usb-background-20261106/dashboard-check.json`.
