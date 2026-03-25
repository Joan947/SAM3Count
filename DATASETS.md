# Datasets 
We evaluate SAM3Count on a diverse set of image and video benchmarks. 
### Image datasets 
* **FSCD-147 / FSC-147** — few-shot style counting benchmark with strong density variation.
* **ShanghaiTech Part A / Part B** — crowd counting benchmarks covering highly dense and relatively sparse scenes.
* **CARPK** — car counting benchmark for overhead imagery.
* **OmniCount-191** — additional image-counting benchmark used in our evaluation scripts.
  
### Video datasets
* **TAO-Count** — large-scale open-vocabulary video counting benchmark. 


# Dataset Preparation

## 1. FSCD-147 

* Download the [FSCD-147](https://drive.google.com/file/d/1sQwTeTGECpIPcbOu8Qnrq29BXwx5IlGF/view?usp=drive_link) images and annotations for the val and test split annotations.
* Organise files as follows:

```bash
data/FSCD-147/
├── images_384_VarV2/
├── annotation_FSC147_384.json
├── instances_test.json
└── instances_val.json
```


## 2. ShanghaiTech

* Download the [ShangaiTech](https://www.kaggle.com/datasets/tthien/shanghaitech) dataset
* Preprocess the data as follows:
```bash
# Navigate to the part_A folder and create directory gt_json
cd partA 
mkdir gt_json
# run the mat_to_json.py file to convert the annotations to a json format
python mat_to_json.py --input_dir ../data/ShanghaiTech/part_A/test_data/ground-truth --output_dir ../data/ShanghaiTech/part_A/test_data/gt_json --mode per_file 
# repeat the same for part_B

cd partB
mkdir gt_json
# run the mat_to_json.py file to convert the annotations to a json format
python mat_to_json.py --input_dir ../data/ShanghaiTech/part_B/test_data/ground-truth --output_dir ../data/ShanghaiTech/part_B/test_data/gt_json --mode per_file
    
```
* Our already preprocessed data is available [here](https://drive.google.com/file/d/1-22swHKg498qvCKan8YIBHdJMh0Fmm06/view?usp=drive_link)
* Organised files as follows:
```bash
data/ShanghaiTech/
├── part_A/
│   └── test_data/
│       ├── images/
│       └── gt_json/
└── part_B/
    └── test_data/
        ├── images/
        └── gt_json/
```


## 3. CARPK

* Download the [CARPK](https://lafi.github.io/LPN/) dataset
* Follow the instructions to obtain the password to unzip the file
* Organised files as follows:

```bash
data/CARPK/
└── data/
    ├── Images/
    ├── Annotations/
    └── ImageSets/
```


## 4. OmniCount-191

* Download the [OmniCount](https://mondalanindya.github.io/OmniCount/) dataset.
* Since we tested on Fruits split make sure it looks like this:

```bash
data/OmniCount-191/
└── Fruits/
    └── test/
        ├── _annotations.coco.json
        └── .... # images
```


## 5. TAO-Count

* Download the TAO validation videos from [here](https://example.com/tao-count-download). You can also follow the directions at this link to download the TAO-Count videos if the first link does not work for you.

### Expected structure

```bash
data/VideoCount/
└── TAO-Count/
    ├── anno/
    │   └── TAO-count-gt.json
    └── frames/
```


## 6. Penguins

Download the Penguins dataset from [here](https://example.com/penguins-download). You can also follow the instructions from the CountVid repository to download the TAO-Count and Penguins video dataset if the first links provided does not work for you.


```bash
data/VideoCount/
└── Penguins/
    ├── anno/
    │   └── penguins-count-gt.json
    └── frames/
```

