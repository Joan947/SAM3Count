# [CVPRW' 26 Oral] SAM3Count: Zero-Shot Open Vocabulary Counting in Images and Videos

**Joana Konadu Owusu, Shivanand Venkanna Sheshappanavar**
Geometric Intelligence Research Lab, University of Wyoming

Official repository for **SAM3Count**, a text-prompted open-vocabulary counting framework for **images** and **videos** built on top of **SAM3**.

> SAM3Count extends SAM3 with two task-specific modules:
> **(1)** an adaptive ROI-guided tiling pipeline for dense image counting, and
> **(2)** a lightweight multi-modal re-identification tracker for video counting.



## Highlights

* **CVPRW 2026 Oral** paper on zero-shot open-vocabulary counting in images and videos.
* **Text-only interface** for counting arbitrary object categories without manual exemplars.
* **Image counting** via density-aware ROI-guided adaptive tiling for dense scenes.
* **Video counting** via a SAM3-based re-identification tracker that reduces ID fragmentation, identity switches, and double counting.
* Strong results on **FSCD-147, ShanghaiTech, CARPK, PixMo, CountBench, TAO-Count, and Penguins**.



## Overview

**SAM3Count** builds on the segmentation and tracking capabilities of **SAM3** and introduces two additions tailored to counting:

* **Images:** a two-stage density-aware pipeline that uses a full-image SAM3 pass for sparse scenes and triggers **ROI-guided adaptive tiling** for dense scenes.
* **Videos:** a **multi-modal re-identification tracker** that maintains a consistent identity space on top of SAM3’s raw track IDs using appearance, motion, spatial, and temporal cues.



## Architecture

### SAM3Count for Images

<img width="1320" height="864" alt="image_archi_sam3count" src="https://github.com/user-attachments/assets/b026c1b9-81ed-4772-b7c9-7e69368aaab4" />

The image pipeline operates in two stages:

#### Stage 1: Full-image inference and density decision

#### Stage 2: ROI-guided adaptive tiling

### SAM3Count for Videos

<img width="1408" height="869" alt="video_archi_sam3count" src="https://github.com/user-attachments/assets/b72cd55a-8a8e-47a9-8884-fa9e2905ac1c" />

The video pipeline augments SAM3 with a lightweight **multi-modal re-identification tracker** which is robust to occlusions, object re-entry, and SAM3 ID fragmentation.



## Installation

SAM3Count depends on the official **SAM3** repository and checkpoint. The setup below follows the same style as a SAM3-based installation.

### 1. Create the environment

```bash
conda create -n sam3count python=3.12 -y
conda activate sam3count

```

### 2. Clone the official SAM3 repository

Place the official SAM3 repository inside the project root so that the folder name is exactly `sam3`. 

```bash
git clone https://github.com/facebookresearch/sam3.git
```
Follow the installation instructions of SAM3 [here](https://github.com/facebookresearch/sam3/blob/main/README.md)

### 3. Download checkpoints

Download the official SAM3 checkpoint [here](https://huggingface.co/facebook/sam3/blob/main/sam3.pt) and the fine tuned checkpoint and place it inside the `checkpoints` directory.

```bash
mkdir -p checkpoints
```

Fine-tuned SAM3Count checkpoint for dense image counting is found [here](https://drive.google.com/file/d/PLACEHOLDER/view?usp=sharing)




## Repository Structure

```bash
SAM3Count/
├── sam3count_images.py   # image counting
├── sam3count_videos.py   # video counting
├── checkpoints/    # local weights
├── scripts/
├── images/
├── videos/
├── outputs/    # saved predictions / visualizations
├── data/    # datasets
├── sam3/    # official SAM3 repo
└── README.md
```



## Demo

### Image demo

```bash
python sam3count_images.py --image_path images/demo.jpg --input_text "bird" --save_vis --show_id
```

### Video demo

```bash
python sam3count_videos.py --video_dir video --input_text "car" --output_dir outputs 
```


## Reproducing Results

### PixMo evaluation

```bash
python scripts/evaluate_pixmo.py --benchmark_file data/pixmo_test.json --output_file outputs/pixmo/predictions.json --metrics_file outputs/pixmo/metrics.json \

```

### FSCD-147 evaluation

```bash
python scripts/evaluate_fscd147.py --json_path fscd147/instances_val_remapped.json --images_dir data/FSC147_384_V2/images_384_VarV2 --num_gpus 2 
```

### OmniCount evaluation

```bash
python scripts/evaluate_omnicount.py --json_path data/OmniCount-191/Fruits/test/_annotations.coco.json --images_dir data/OmniCount-191/Fruits/test --num_gpus 4 
```

### ShanghaiTech evaluation

Part A
```bash
python scripts/evaluate_shanghaitech.py --ann_json_dir data/ShanghaiTech/part_A/test_data/gt_json --images_dir data/ShanghaiTech/part_A/test_data/images --num_gpus 4 --output_json outputs/shanghai/partA_results.json --summary_json outputs/shanghai/partA_summary.json --temp_dir ./temp_shanghai_partA_test

```
Part B

```bash
python scripts/evaluate_shanghaitech_multi_gpu.py \
  --ann_json_dir data/ShanghaiTech/part_B/test_data/gt_json --images_dir data/ShanghaiTech/part_B/test_data/images --num_gpus 4 --output_json outputs/shanghai/partB_results.json --summary_json outputs/shanghai/partB_summary.json --temp_dir ./temp_shanghai_partB_test 
```

### CARPK evaluation

```bash
python scripts/evaluate_carpk.py --dataset_root data/CARPK/data --split test --num_gpus 2 --output_json outputs/carpk/results.json --summary_json outputs/carpk/summary.json

```

### TAO-Count evaluation

```bash
python scripts/evaluate_tao_count.py --output_file outputs/tao_count/predictions.json --data_dir data/VideoCount/TAO-Count --downsample_factor 2.0

python scripts/evaluate_counting.py --ground_truth data/VideoCount/TAO-Count/anno/TAO-count-gt.json --predicted outputs/tao_count/predictions.json  --parent_dir data/VideoCount/TAO-Count/frames
```



## Datasets

Following prior open-vocabulary counting work, we evaluate SAM3Count on a diverse set of image and video benchmarks.

### Image datasets

* **FSCD-147 / FSC-147** — few-shot style counting benchmark with strong density variation.
* **ShanghaiTech Part A / Part B** — crowd counting benchmarks covering highly dense and relatively sparse scenes.
* **CARPK** — car counting benchmark for overhead imagery.
* **PixMo-Count** — open-world counting benchmark with natural-language prompts.
* **CountBench** — benchmark for counting under open-vocabulary prompt settings.
* **OmniCount-191** — additional image-counting benchmark used in our evaluation scripts.

### Video datasets

* **TAO-Count** — large-scale open-vocabulary video counting benchmark.

Please follow the official dataset websites, licenses, and terms of use when downloading and using the data.

A recommended layout is:

```bash
data/
├── FSCD-147/
├── ShanghaiTech/
├── CARPK/
├── OmniCount-191/
└── VideoCount/
    └── TAO-Count/
```



## Results

### Dense image benchmarks

| Method         | Benchmark           | MAE↓   | RMSE↓  |
| -------------- | ------------------- | ------ | ------ |
| SAM3Count      | FSCD-147 Val        | 25.02  | 81.28  |
| SAM3Count (ft) | FSCD-147 Val        | 14.34  | 59.04  |
| SAM3Count      | FSCD-147 Test       | 19.80  | 128.59 |
| SAM3Count (ft) | FSCD-147 Test       | 13.02  | 86.42  |
| SAM3Count      | ShanghaiTech Part A | 213.75 | 376.40 |
| SAM3Count (ft) | ShanghaiTech Part A | 131.27 | 262.82 |
| SAM3Count      | ShanghaiTech Part B | 36.43  | 56.97  |
| SAM3Count (ft) | ShanghaiTech Part B | 28.88  | 44.37  |


### Domain-specific image benchmarks

| Method    | Benchmark       | MAE↓ | RMSE↓ |
| --------- | --------------- | ---- | ----- |
| SAM3Count | CARPK           | 3.11 | 5.60  |
| SAM3Count | OmniCount-Fruit | TBD  | TBD   |

> **Note:** OmniCount-Fruit results will be added once we finalize and verify the evaluation numbers for the released setup.

### Video benchmarks

| Method    | Benchmark | MAE↓ | RMSE↓ |
| --------- | --------- | ---- | ----- |
| SAM3Count | TAO-Count | 0.78 | 1.63  |

## Notes on Zero-Shot vs Fine-Tuned Results

* **SAM3Count** refers to the **training-free, text-only** setting.
* **SAM3Count (ft)** refers to a **fine-tuned** variant used primarily for dense image benchmarks.

When comparing against prior methods, please note that some baselines use **exemplars** while SAM3Count is designed around a **text-only** interface.



## Citation

If you find this repository useful, please cite our paper.




## Acknowledgements

This repository builds on the official **SAM3** codebase and the recent literature on open-vocabulary counting in images and videos.

We thank the open-source and research communities whose work made this project possible.


## License

Please add the intended license for this repository here.
