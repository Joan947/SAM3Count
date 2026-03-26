
# [CVPRW' 26 Oral] SAM3Count: Zero-Shot Open Vocabulary Counting in Images and Videos
<div>
  <strong>Joana Konadu Owusu &amp; Shivanand Venkanna Sheshappanavar</strong><br>
  Geometric Intelligence Research Lab, University of Wyoming
</div>

---

## Highlights

* **SAM3Count** has been accepted at WiCV @ CVPR'26 and selected for Oral presentation.
* Full fine-tuning code and other dataset evaluations will be released soon.
* **Inference** and **demo** code for some of the evaluation benchmarks has been released.
  

## Overview
**SAM3Count**, is a text-prompted open-vocabulary counting framework for **images** and **videos** built on top of **SAM3**.

It introduces two additions tailored to counting:

* **Images:** a two-stage density-aware pipeline that uses a full-image SAM3 pass for sparse scenes and triggers **ROI-guided adaptive tiling** for dense scenes.
* **Videos:** a **multi-modal re-identification tracker** that maintains a consistent identity space on top of SAM3’s raw track IDs using appearance, motion, spatial, and temporal cues.



## Architecture

### SAM3Count for Images

<img width="1320" height="864" alt="image_archi_sam3count" src="https://github.com/user-attachments/assets/b026c1b9-81ed-4772-b7c9-7e69368aaab4" />

The image pipeline operates in two stages:

* Stage 1: Full-image inference and density decision

* Stage 2: ROI-guided adaptive tiling

### SAM3Count for Videos

<img width="1408" height="869" alt="video_archi_sam3count" src="https://github.com/user-attachments/assets/b72cd55a-8a8e-47a9-8884-fa9e2905ac1c" />

* The video pipeline augments SAM3 with a lightweight **multi-modal re-identification tracker** which is robust to occlusions, object re-entry, and SAM3 ID fragmentation.



## Installation

SAM3Count depends on the official **SAM3** repository and checkpoint. The setup below follows the same style as a SAM3-based installation.

### 1. Create the environment

```bash
conda create -n sam3count python=3.12 -y
conda activate sam3count
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 2. Clone the official SAM3 repository

Place the official SAM3 repository inside the project root so that the folder name is exactly `sam3`. 

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
pip install -e ".[notebooks]"
pip install -e ".[train,dev]"
```
Follow the installation instructions of SAM3 [here](https://github.com/facebookresearch/sam3/blob/main/README.md)
and check the official repo for any FAQS with regard to SAM3.

### 3. Download checkpoints

Download the official SAM3 checkpoint [here](https://huggingface.co/facebook/sam3/blob/main/sam3.pt) and the fine tuned checkpoint and place it inside the `checkpoints` directory.

```bash
mkdir -p checkpoints
```

Fine-tuned SAM3Count checkpoint for dense image counting is found [here](https://drive.google.com/file/d/1uSpEeM9l9SN_Q_W-CJqYy75mQ-O75HpL/view?usp=drive_link)




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
python sam3count_images.py --image_path examples/images/demo.jpg --input_text "bird" --save_vis --show_id
```

### Video demo

```bash
python sam3count_videos.py --video_dir examples/video --input_text "car" --output_dir outputs 
```


## Reproducing Results

<!-- ### PixMo evaluation

```bash
python scripts/evaluate_pixmo.py --benchmark_file data/pixmo_test.json --output_file outputs/pixmo/predictions.json --metrics_file outputs/pixmo/metrics.json 

``` -->

<!-- ### FSCD-147 evaluation

```bash
python scripts/evaluate_fscd147.py --json_path fscd147/instances_val_remapped.json --images_dir data/FSC147_384_V2/images_384_VarV2 --num_gpus 2 
``` -->
<!-- 
### OmniCount evaluation

```bash
python scripts/evaluate_omnicount.py --json_path data/OmniCount-191/Fruits/test/_annotations.coco.json --images_dir data/OmniCount-191/Fruits/test --num_gpus 4 
``` -->

### ShanghaiTech evaluation

Part A
```bash
python scripts/evaluate_shanghaitech.py --ann_json_dir data/ShanghaiTech/part_A/test_data/gt_json --images_dir data/ShanghaiTech/part_A/test_data/images --num_gpus 4 --output_json outputs/shanghai/partA_results.json --summary_json outputs/shanghai/partA_summary.json --temp_dir ./temp_shanghai_partA_test

```
Part B

```bash
python scripts/evaluate_shanghaitech.py --ann_json_dir data/ShanghaiTech/part_B/test_data/gt_json --images_dir data/ShanghaiTech/part_B/test_data/images --num_gpus 4 --output_json outputs/shanghai/partB_results.json --summary_json outputs/shanghai/partB_summary.json --temp_dir ./temp_shanghai_partB_test 
```

### CARPK evaluation

```bash
python scripts/evaluate_carpk.py --dataset_root data/CARPK/data --split test --num_gpus 2 --output_json outputs/carpk/results.json --summary_json outputs/carpk/summary.json

```

<!-- ### TAO-Count evaluation

```bash
python scripts/evaluate_tao_count.py --output_file outputs/tao_count/predictions.json --data_dir data/TAO-Count --downsample_factor 2.0

python scripts/evaluate_counting.py --ground_truth data/VideoCount/TAO-Count/anno/TAO-count-gt.json --predicted outputs/tao_count/predictions.json  --parent_dir data/VideoCount/TAO-Count/frames
``` -->

### Penguin evaluation
```bash
python scripts/eval_penguins.py --output_file outputs/penguin/predictions.json --data_dir data/Penguins 

python evaluate_counting_accuracy.py --ground_truth data/Penguins/anno/penguins-count-gt.json --predicted outputs/penguin/predictions.json --parent_dir data/Penguins/frames
```

## Datasets
For dataset download and preprocessing instructions, please see [DATASETS.md](DATASETS.md).

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
| SAM3Count | OmniCount-Fruit | 0.43 | 0.93  |


### Video benchmarks

| Method    | Benchmark | MAE↓ | RMSE↓ |
| --------- | --------- | ---- | ----- |
| SAM3Count | TAO-Count | 0.78 | 1.63  |
| SAM3Count | Penguin   |  2.3 |  3.1  |

* **SAM3Count** refers to the training-free, setting.
* **SAM3Count (ft)** refers to a **fine-tuned** variant used primarily for benchmarks (FSCD147 and ShangaiTech) with dense scenes.



## Citation

If you find SAM3Count please cite our work and give it a :star:


## Acknowledgements

SAM3Count is built on top of **[SAM3](https://github.com/facebookresearch/sam3)** by Meta FAIR and the recent literature on open-vocabulary counting in images and videos. And we used VideoCount from **[CountVID](https://github.com/niki-amini-naieni/CountVid)** to test SAM3Count's video counting capabilities.

