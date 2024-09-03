# Introduction

PyTorch implementation of paper:

FAWL: Weakly-Supervised Video Corpus Moment Retrieval with Frame-Wise Auxiliary Alignment and Weighted Contrastive Learning

> The codes are modified from [Code-JSG](https://https://github.com/CFM-MSG/Code_JSG)


# Environment Setup

```bash
conda create -n fawl
conda activate fawl
pip install -r requirements.txt
```

# Data Preparation

* For Charades-STA, follow [Code-JSG](https://https://github.com/CFM-MSG/Code_JSG) to prepare all features and annotations.
* For ActivityNet-Captions, follow [MS-SL](https://github.com/HuiGuanLab/ms-sl) to prepare required features. 
  * Use the script `utils/convert_hdf5.py` to convert downloaded features to `activitynet_i3d.hdf5` file. Make sure to replace paths in the script correctly according to downloaded ms-sl files.
  * We provide converted annotations in `data/activitynet/TextData`.

* The final directory structure is expected as:

  * ```
    data
    |-- charades
    |   |-- TextData
    |   |-- charades_i3d_rgb_lgi.hdf5
    |   |-- results  # containing provided checkpoint
    |-- activitynet
    |   |-- TextData
    |   |-- activitynet_i3d.hdf5
    |   |-- results  # containing provided checkpoint
    ```



# Training and Inference

```bash
#Add project root to PYTHONPATH (Note that you need to do this each time you start a new session.)
source setup.sh

## Training
# Optional: change exp_id, device_ids in scripts/train_*.sh

# for Charades-STA
bash scripts/train_charades.sh

# for Activitynet-Captions
bash scripts/train_activitynet.sh
# The model is placed in the directory data/$collection/results/VCMR/$ckpt_name after training

## Inference
# set model_dir to the checkpoint direction in scripts/test.sh correctly
bash scripts/test.sh
```

