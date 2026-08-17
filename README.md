# CRM: Context-aware Refinement Module

Training code for self-supervised depth estimation in transparent environments.
This repository accompanies **Knowledge Distillation with Context-aware
Refinement for Self-Supervised Depth Estimation in Transparent Environments**.

<div align="left">
  <img src="images/Overview.png" alt="CRM overview" width="800" />
</div>

## JBNU stereo: environment setup on this server

The commands in this section target the current machine:

- Ubuntu/Linux, NVIDIA driver `565.57.01`
- 5 x NVIDIA RTX 6000 Ada (48 GB)
- system CUDA Toolkit `11.8`
- project path `/ssd1/jm_data/ijcas/sqldepth_crm`

The checked-in `requirements.txt` is an archive of the original environment.
Do **not** install it directly on this server: its PyTorch entries point to
nonexistent Python 3.7 wheel files on another machine, and some of its package
pins are mutually incompatible with Python 3.10. Use the clean environment
below instead.

### 1. Create a Conda environment

```bash
source /home/milab/miniconda3/etc/profile.d/conda.sh
conda create -n crm python=3.10 pip -y
conda activate crm
python -m pip install --upgrade pip setuptools wheel
```

Install a CUDA 11.8 PyTorch build. The PyTorch wheel contains the CUDA runtime;
it does not require changing the server's system CUDA installation.

```bash
python -m pip install \
  torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu118
```

Install the packages needed by the teacher and student training code:

```bash
python -m pip install \
  numpy==2.2.6 \
  pillow==11.3.0 \
  opencv-python-headless==4.12.0.88 \
  scipy==1.15.3 \
  scikit-image==0.25.2 \
  matplotlib==3.10.5 \
  tqdm==4.67.1 \
  tensorboard==2.20.0 \
  timm==1.0.19 \
  kornia==0.8.1
```

Only `opencv-python-headless` is needed for training. Do not install
`opencv-python` alongside it unless GUI functions such as `cv2.imshow` are
required.

### 2. Verify the installation

Run all project commands from the repository root because the argument files
contain relative paths. The student trainer also imports the sibling module
`/ssd1/jm_data/ijcas/crm`.

```bash
cd /ssd1/jm_data/ijcas/sqldepth_crm

test -d /ssd1/jm_data/ijcas/data/jbnu_stereo
test -f pretrained/resnet_320x1024/models/weights_24/encoder.pth
test -f ../crm/module.py

PYTHONPATH=/ssd1/jm_data/ijcas python -c '
import torch, torchvision, cv2, kornia, skimage, timm
from trainer import Trainer
from trainer_teacehr import TeacherTrainer
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
'
```

`CUDA available` must be `True`. If it is `False`, first check that the job or
shell has access to a GPU with `nvidia-smi`.

## JBNU stereo training

The intended order is teacher first, then CRM student. Start with one GPU; the
configured batch size of 8 fits an RTX 6000 Ada. Select another GPU by changing
`CUDA_VISIBLE_DEVICES`.

### 1. Train the teacher

```bash
cd /ssd1/jm_data/ijcas/sqldepth_crm
conda activate crm

CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/ssd1/jm_data/ijcas \
python train.py \
  args_files/jm/jbnu_stereo/jbnu_stereo_352x640_train_teacher.txt
```

The teacher argument file currently writes checkpoints below:

```text
logs/revision/jbnu_stereo_resnet_352x640_teacher_resnet_320x1024_weights_24/models/weights_<epoch>
```

Training uses zero-based epoch numbers, so a completed 25-epoch run normally
has `weights_24`. Confirm that it contains at least `encoder.pth` and
`depth.pth`.

### 2. Point the student at the trained teacher

Before starting the student, set `--teacher_path` in
`args_files/jm/jbnu_stereo/jbnu_stereo_352x640_train.txt` to the teacher
checkpoint to use. For a freshly completed run, that is normally:

```text
--teacher_path /ssd1/jm_data/ijcas/sqldepth_crm/logs/revision/jbnu_stereo_resnet_352x640_teacher_resnet_320x1024_weights_24/models/weights_24
```

The file currently points to
`logs/revision/teacher/models/weights_19`, which already exists but is not the
output directory of the teacher command above. Leave it unchanged only when
that existing checkpoint is intentionally being used.

### 3. Train the student

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/ssd1/jm_data/ijcas \
python train.py \
  args_files/jm/jbnu_stereo/jbnu_stereo_352x640_train.txt
```

Logs and student checkpoints are written below
`logs/revision/ablation/consistency_mask_001/jbnu_stereo_consistency`.

To keep a long run alive after disconnecting from SSH, execute the same command
inside `tmux` or `screen`. TensorBoard can be started with:

```bash
tensorboard --logdir /ssd1/jm_data/ijcas/sqldepth_crm/logs --port 6006
```

## Dataset preparation

The JBNU argument files expect the dataset at:

```text
/ssd1/jm_data/ijcas/data/jbnu_stereo
```

Training split files are under `splits/jbnu_stereo`. Transparent-object masks
must already be present in the layout expected by the dataset loader. The
original project used
[Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) to
prepare these masks; Grounded-SAM is a separate preprocessing environment and
is not required to launch training when masks have already been generated.

## Citation

Citation information will be added when the paper is published.
