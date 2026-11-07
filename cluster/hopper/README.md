# Train TurtleHero on GMU Hopper

This workflow prepares existing RGB and synthetic RGB-depth training for one NVIDIA GPU. It rebuilds dataset paths on Hopper, verifies inputs, records each job's environment, and writes checkpoints to scratch. No connection, upload, or cluster job is performed by installing these files. `submit` prints the command unless `--execute` is supplied.

## 1. Prepare locally when the data is ready

From the TurtleHero project directory:

```sh
.venv/bin/python cluster/hopper/hopper.py inventory \
  --manifest datasets/turtle-gaussian-v1/manifest.json \
  --checkpoint models/challenge-20261019/turtle.pt \
  --output output/hopper/transfer-files.txt
```

Inspect `output/hopper/transfer-files.txt` and its adjacent `.audit.json`. The inventory includes selected images, labels, depth profiles, render metadata, the manifest, initialization weights, code, and training tests. It excludes virtual environments, unrelated model/run folders, and raw camera sessions unless a selected manifest explicitly references their images. It creates a list, not a duplicate dataset archive. Regenerate it after changing code or inputs.

Future images need reviewed YOLO labels and a new manifest. Use the existing manifest as the schema: a JSON array of project-relative `image` and `label` paths, `split` (`train`, `val`, or `test`), `sha256`, `label_sha256`, `width`, `height`, and `boxes`. Use matching `datasets/NAME/images/SPLIT/FILE.jpg` and `datasets/NAME/labels/SPLIT/FILE.txt` paths. An empty reviewed label file represents a verified background image. Do not turn an unreviewed capture into a negative. Add `reviewed: true`, source/rights, and `session_id`, `individual_id`, or `observation_id` when known. Keep an encounter/individual and all its derivatives in one split; reserve independent real encounters for testing. Hash checks do not detect different photographs of the same individual when group IDs are missing.

The audit rejects changed hashes, missing files, invalid labels, explicit unreviewed/review-only inputs, identical images across splits, and shared group IDs across splits. It reports missing group metadata. It does not create annotations or infer biological identity. Keep datasets immutable while jobs are queued/running.

**More real RGB photos:** set `mode` to `rgb` and point `manifest` to the new reviewed dataset. **Fusion:** the current adapter still trains only the synthetic rows with explicitly simulated `oak` and `iphone` geometry profiles and 640-square images. Real RGB rows initialize the detector through a separately trained RGB checkpoint; they are not paired fusion samples. Collected real RGB-depth needs a reviewed paired-data adapter before fused training can consume it. The new photo-fitted shell examples remain review-only and are intentionally rejected. This deployment does not silently promote them into training.

## 2. Transfer only when ready

Obtain a GMU ORC account and connect using the documented [Hopper SSH/VPN/Duo workflow](https://wiki.orc.gmu.edu/mkdocs/Logging_Into_Hopper/). On your Mac, replace `your_netid`:

```sh
GMU_ID=your_netid
ssh "$GMU_ID@hopper.orc.gmu.edu" "mkdir -p /scratch/$GMU_ID/TurtleHero"
rsync -av --dry-run --files-from=output/hopper/transfer-files.txt ./ \
  "$GMU_ID@hopper.orc.gmu.edu:/scratch/$GMU_ID/TurtleHero/"
# After reviewing the dry run, remove --dry-run to transfer these files.
```

These commands are for you to run when you authorize moving the selected data to GMU. No transfer has been performed. Do not rsync the entire home directory or copy `.venv`, `.venv-oak`, or `.venv-coreml`. This machine's Git root may be outside TurtleHero, so do not use a parent-level Git archive as the deployment package.

## 3. Create the Linux environment on Hopper

```sh
cd /scratch/$USER/TurtleHero
module spider python
# Load an available CPython 3.12 module and its required compiler modules.
# Use the exact module names shown by Hopper; keep them loaded when submitting jobs.
python3 --version
bash cluster/hopper/setup.sh
```

If the loaded interpreter has another name, use `PYTHON_BIN=/path/to/python3.12 bash cluster/hopper/setup.sh`. Setup checks Linux x86_64 and Python 3.12, creates `/scratch/$USER/turtlehero/env`, installs PyTorch 2.14.0 + torchvision 0.29.0 from the official CUDA 12.6 wheel index, then installs pinned training packages and records `environment.lock.txt`. Existing environments are never modified; choose a new destination argument to rebuild. Dependency installation uses the network. No datasets or checkpoints are downloaded by this workflow.

Linux CPython 3.12 wheels for these versions were checked on 2026-10-26. CUDA execution and Hopper's installed driver/modules still require the smoke job below. If setup or `cv2` import fails due to a missing system library, use the appropriate Hopper module or ask ORC support; do not mix a second PyTorch installation into this environment. The new environment is separate from `pyproject.toml` and the Mac capture environment. Setup is not an exhaustive transitive lock; it saves the resolved package versions for each installation and job.

## 4. Configure and submit

```sh
cp cluster/hopper/config.example.json cluster/hopper/config.json
# Edit config.json: run name, manifest, checkpoint, mode, epochs, batch, and Slurm account if required.
PY=/scratch/$USER/turtlehero/env/bin/python
$PY cluster/hopper/hopper.py submit --config cluster/hopper/config.json --smoke
$PY cluster/hopper/hopper.py submit --config cluster/hopper/config.json --smoke --execute
```

The example reproduces a fresh fusion run initialized from the original P2 RGB detector, with 40 epochs and batch 8. For more RGB photos, change `mode` to `rgb`, retain `variant: "p2"` for that P2 checkpoint, and use `learning_rate: 0.0003` as the existing RGB recipe's starting point. For a new fusion experiment after RGB fine-tuning, point `checkpoint` to that RGB run's P2 `weights/best.pt`. Do not pass a fused inference checkpoint as the RGB initialization.

The default request is `gpuq`, QoS `gpu`, `gpu:A100.80gb:1`, 8 CPUs, 32 GB host memory, and 8 hours. Set your account if required by your allocation. Partition, QoS, GPU type, resources, and paths are editable. These defaults follow [GMU's GPU documentation](https://wiki.orc.gmu.edu/mkdocs/Running_GPU_Jobs/); full A100/H100 jobs currently require batch submission, while interactive jobs use MIG. Contributor partitions may preempt jobs. This trainer is single-GPU; it does not implement distributed training. Batch 8 is a starting point, not a measured throughput optimum.

The smoke job requests 15 minutes, checks GPU visibility, CUDA forward/backward and torchvision NMS, verifies the dataset and fusion input shapes, then performs one epoch (two batches for fusion). It uses a distinct run name and does not replace existing models. Check `logs/` and the smoke run's `completed.json`. After it passes:

```sh
$PY cluster/hopper/hopper.py submit --config cluster/hopper/config.json --execute
squeue -u "$USER"
# After noting the numeric job ID:
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

Without `--execute`, submission only prints a shell-quoted command. GPU work is gated behind a Slurm allocation and fails if CUDA is unavailable; it never falls back to CPU on a login node. Runtime settings disable experiment uploads and use per-job configuration/cache directories. A lock prevents simultaneous jobs from writing the same run.

## Outputs and restart

By default, all job output is below `/scratch/$USER/turtlehero/`:

| Path | Contents |
| --- | --- |
| `logs/turtle-NAME-JOBID.out` | Slurm stdout/stderr |
| `runtime/JOBID/` | Config, exact command, package versions, GPU/driver and preflight results |
| `prepared/NAME/` | Locally rebuilt dataset YAML/split lists, manifest and checksum audit |
| `runs/fusion/NAME/` | `turtle_fused.pt`, `training_state.pt`, provenance, history, progress and completion |
| `runs/rgb/NAME/` | Ultralytics `weights/best.pt`, `weights/last.pt`, metrics and provenance |

After an interruption, wait until the previous job has exited, keep the same config/code/data paths, then:

```sh
$PY cluster/hopper/hopper.py submit --config cluster/hopper/config.json --resume --execute
```

Fusion restarts from the last completed epoch with model, optimizer, scheduler, data-loader RNG and Torch/CUDA RNG state. At most the unfinished epoch is lost. It validates source, input checksums, and recipe; changing total epochs is a new experiment, not a resume. Older inference-only packages cannot resume optimizer state. A timeout before the first checkpoint requires a fresh run name. RGB restart uses Ultralytics `last.pt` and its original recipe. Do not move its dataset/run paths before resuming. No automatic requeue is enabled.

Scratch is temporary: [GMU documents a 90-day purge](https://wiki.orc.gmu.edu/mkdocs/Running_GPU_Jobs/). After jobs finish, archive runs, runtime records, manifests, and input hashes in your allocated persistent project storage or download them. For example, after selecting a run on Hopper:

```sh
rsync -av /scratch/$USER/turtlehero/runs/fusion/fused-more-data-v1/ \
  /projects/YOUR_PROJECT/turtlehero/fused-more-data-v1/
```

Keep original labeled data and its manifest persistently as well. Do not treat scratch or a model checkpoint as a data backup. Follow your allocation's storage policy.

## Scope and verification

The Linux workflow supports detector training using prepared RGB and synthetic RGB-depth assets. Existing Gaussian images/depth can train on Hopper. `metalsplat` remains Mac/Metal-only here; the separate experimental Nerfstudio container job below prepares CUDA reconstruction, while Blender cluster jobs remain separate work. OAK capture stays local. No actual Hopper account, driver, CUDA execution, remote upload, or Slurm submission has been tested in this session.

Local checks cover dataset relocation, checksum/split/review guards, epoch restart state, existing geometry/fusion tests, shell syntax and dry-run submission. Small CPU fusion runs verify the new training checkpoint format without changing any packaged model. Run `python -m unittest discover -s tests/training -v` in a training environment for these checks. Hardware validation is the submitted smoke job, followed by independently labeled real evaluation; a successful training job is not proof of field accuracy.

## Construction-site Gaussian reconstruction candidates

The imported `What About The Turtles Photos` archive provides two initial video candidates in `datasets/site-candidates/`: `arrowbrook-20211021` (40 frames, original seconds 1–21) and `hartland-20240428` (36 frames, seconds 11–29). These short segments were selected from contact sheets. Parallax and recoverable camera poses remain unverified; Arrowbrook includes moving equipment and Hartland's later human close-ups were excluded. Water and moving objects in other videos make them weaker static-scene inputs. Do not merge different dates/sites into one reconstruction.

To reproduce a candidate locally:

```sh
.venv/bin/python prepare_site_scene.py --video-id 0067 --start 1 --end 21 \
  --output datasets/site-candidates/arrowbrook-new
.venv/bin/python cluster/hopper/hopper.py scene-inventory \
  --scene datasets/site-candidates/arrowbrook-20211021 datasets/site-candidates/hartland-20240428 \
  --output output/hopper/scene-transfer-files.txt
```

Use that file list with the same reviewed `rsync --files-from` procedure above. It includes selected frames, their original PTS/checksums/source metadata and cluster scripts, not the entire archive. Original videos remain local and intact. Intrinsics are estimated by SfM for the actual video; the camera model name alone is not a calibration. No measured depth or metric scale is supplied.

On Hopper, load its Apptainer or Singularity module (`module spider apptainer` / `module spider singularity`). Then:

```sh
bash cluster/hopper/setup_splat.sh
ROOT=/scratch/$USER/TurtleHero
SIF=/containers/dgx/UserContainers/$USER/nerfstudio-1.1.3.sif
sbatch cluster/hopper/splat_job.sh "$ROOT" "$SIF" \
  "$ROOT/datasets/site-candidates/arrowbrook-20211021" \
  "/scratch/$USER/turtlehero/scenes/arrowbrook-first"
```

Add `--account=YOUR_ACCOUNT` before the script name if your allocation requires it. Setup downloads a large official Nerfstudio container **on Hopper**, separate from the detector environment. Version 1.1.3 is pinned by registry digest `sha256:e0da96a781c45b6bf46d08774a50dd8785c4d63be0a033c20e31bbce3e9a5121`, verified against GHCR on 2026-10-26; no container was downloaded locally. The default image location follows [GMU container guidance](https://wiki.orc.gmu.edu/mkdocs/Containerized_jobs_on_Hopper/). Pass a different authorized destination to setup if that location is unavailable. Native Docker is not required.

The job verifies frame hashes, runs COLMAP through `ns-process-data`, and requires at least 20 poses and 80% registration before running 30,000 Splatfacto iterations and exporting a Gaussian PLY. These are initial screening thresholds, not proof of accurate geometry. The commands follow [Nerfstudio's image-processing workflow](https://docs.nerf.studio/quickstart/custom_dataset.html) and [Splatfacto documentation](https://docs.nerf.studio/nerfology/methods/splat.html). The image/container integration is untested on Hopper; its first run is an experiment. It records command lines, package versions, container checksum, source metadata and registration counts. A failed registration gate stops before optimization. No scene has been reconstructed yet, and no automatic restart is implemented for this separate scene job.

Inspect recovered cameras, point geometry, floaters and held-out novel views before using a reconstruction as a synthetic background. A nearly stationary pan can have substantial image overlap but insufficient parallax for reliable depth. Add a measured site reference to establish metric scale; rendered splat depth must stay labeled reconstructed/synthetic. Keep the source site/encounter and all its generated turtle composites in one detector split.
