# Precise Liquid Pouring with Imitation Learning

**Physical AI Hackathon 2026** — Teaching robots to pour exactly 50ml using learned policies.

## 🎯 The Challenge

Pouring exactly 50ml — not 40, not 60 — demands real-time perception, compliant motion, and adaptive control that traditional programming can't reliably deliver.

| Metric | Classical Approaches |
|--------|---------------------|
| Success Rate | < 30% |
| Error Margin | ±10ml |
| Setup Time | Hours per container type |

## 🤖 Our Approach

We trained imitation learning policies on teleoperated demonstrations using the **LeRobot** framework.

### Hardware
- **Robot**: SO-ARM101 (6-DOF manipulator)
- **Cameras**: Front + Wrist mounted (640x480 @ 30fps)
- **Training**: Google Colab A100 GPU

### Dataset
- **44 teleoperated episodes** of precise 50ml pours
- **29,250 frames** with dual camera views
- Available on HuggingFace: [`krishnam94/pour_50ml_SO_101`](https://huggingface.co/datasets/krishnam94/pour_50ml_SO_101)

## 🧠 Models Trained

### ACT (Action Chunking Transformer)
Best suited for our dataset size and task complexity.

| Variant | Config | Result |
|---------|--------|--------|
| **Baseline** | chunk_size=50, batch=32, lr=1e-5 | Stable convergence |
| **Large Chunks** | chunk_size=100, batch=64, lr=1e-6 | Similar performance |
| **Robust** ⭐ | kl_weight=5, image_augmentation=on | Best generalization |

**Model**: [`krishnam94/act_pour_50ml`](https://huggingface.co/krishnam94/act_pour_50ml)

### Pi0-FAST (VLA)
Larger foundation model fine-tuned on our task.

**Model**: [`TheLastSid/pi0fast_pour_50ml`](https://huggingface.co/TheLastSid/pi0fast_pour_50ml)

## 🛠️ Setup & Calibration

### 1. Install LeRobot

```bash
# Clone and install LeRobot
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
```

### 2. Calibrate Your Robot

Before collecting data, calibrate both the leader (teleoperation) and follower (robot) arms:

```bash
# Calibrate the follower arm (robot)
lerobot-calibrate --robot.type=so100_follower --robot.port=/dev/tty.usbmodem5A7A0185761

# Calibrate the leader arm (for teleoperation)
lerobot-calibrate --teleop.type=so100_leader --teleop.port=/dev/tty.usbmodem5A7A0185762
```

**Calibration steps:**
1. Move arm to middle of range of motion → press Enter
2. Move each joint through full range → press Enter when done
3. Calibration saves to `~/.cache/huggingface/lerobot/calibration/`

### 3. Test Teleoperation

```bash
lerobot-teleoperate \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem5A7A0185761 \
    --teleop.type=so100_leader \
    --teleop.port=/dev/tty.usbmodem5A7A0185762
```

## 📹 Collecting Training Data

### Record Demonstrations

```bash
lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem5A7A0185761 \
    --robot.cameras='{"wrist": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}, "front": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}}' \
    --teleop.type=so100_leader \
    --teleop.port=/dev/tty.usbmodem5A7A0185762 \
    --dataset.repo_id=YOUR_USERNAME/pour_50ml \
    --dataset.num_episodes=50 \
    --dataset.single_task="Pour 50ml of liquid from cup to target cup" \
    --dataset.push_to_hub=true
```

**Tips for quality demonstrations:**
- Keep movements smooth and consistent
- Aim for 10-15 second episodes
- Vary starting positions slightly between episodes
- Maintain consistent lighting
- We collected **44 episodes** (~30 min total recording time)

### Resume Recording (if interrupted)

```bash
lerobot-record \
    --robot.type=so100_follower \
    ... \
    --dataset.num_episodes=10 \
    --resume=true
```

## 🚀 Training

### Training (Google Colab)

```python
# Install LeRobot
!pip install -q lerobot

# Mount Google Drive for checkpoints
from google.colab import drive
drive.mount('/content/drive')

# Download dataset
!huggingface-cli download krishnam94/pour_50ml_SO_101 \
    --repo-type dataset --local-dir ./dataset

# Train ACT policy
!lerobot-train \
    --dataset.repo_id=krishnam94/pour_50ml_SO_101 \
    --dataset.root=./dataset \
    --policy.type=act \
    --policy.chunk_size=50 \
    --policy.n_action_steps=50 \
    --policy.kl_weight=5 \
    --dataset.image_transforms.enable=true \
    --batch_size=32 \
    --optimizer.lr=1e-5 \
    --steps=40000 \
    --save_freq=5000 \
    --output_dir=/content/drive/MyDrive/act_pour_50ml \
    --wandb.enable=false
```

### Resume Training

```bash
!lerobot-train \
    --config_path=/content/drive/MyDrive/act_pour_50ml/checkpoints/last/pretrained_model/train_config.json \
    --resume=true
```

### Run Inference

```bash
lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem5A7A0185761 \
    --robot.cameras='{"wrist": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}, "front": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}}' \
    --policy.path=~/Desktop/act_40k \
    --dataset.repo_id=krishnam94/eval_pour_50ml \
    --dataset.num_episodes=10 \
    --dataset.single_task="Pour 50ml of liquid" \
    --display_data=true
```

## 📊 Results

| Model | Training Time | Loss | Inference Speed |
|-------|--------------|------|-----------------|
| ACT Baseline | 2.5 hrs | 0.086 | 30 Hz |
| ACT Robust | 2.5 hrs | 0.088 | 30 Hz |
| Pi0-FAST | 8+ hrs | — | ~10 Hz |

### Key Finding
**Lower KL weight (5 vs 10) + image augmentation** produced faster, more confident robot motion that better matched demonstration style while generalizing to lighting variations.

## 📁 Repository Structure

```
├── README.md
├── train_act.py          # ACT training script for Colab
├── run_policy.py         # Local inference script
└── calibration/          # Robot calibration backups
    ├── so_follower_None.json
    ├── so_follower_main.json
    └── so_leader_None.json
```

## 🔧 Troubleshooting

### Find Robot Port
```bash
ls /dev/tty.usb*
```

### Check Camera Indices
```bash
python -c "import cv2; [print(f'Camera {i}: {cv2.VideoCapture(i).isOpened()}') for i in range(5)]"
```

### Restore Calibration
```bash
cp calibration/so_follower_None.json ~/.cache/huggingface/lerobot/calibration/robots/so_follower/None.json
```

### Memory Issues on Mac (Pi0-FAST)
Pi0-FAST requires ~16GB+ RAM. Use ACT for Mac inference — it's faster and works well.

## 🔗 Links

- **Dataset**: [krishnam94/pour_50ml_SO_101](https://huggingface.co/datasets/krishnam94/pour_50ml_SO_101)
- **ACT Model**: [krishnam94/act_pour_50ml](https://huggingface.co/krishnam94/act_pour_50ml)
- **Pi0-FAST Model**: [TheLastSid/pi0fast_pour_50ml](https://huggingface.co/TheLastSid/pi0fast_pour_50ml)
- **LeRobot**: [huggingface/lerobot](https://github.com/huggingface/lerobot)

## 👥 Team

Physical AI Hackathon 2026

## 📄 License

MIT
