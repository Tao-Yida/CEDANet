# Traing of BVM model

This repository contains code from the following two papers:
1. [`Transmission-Guided Bayesian Generative Model for Smoke Segmentation`](https://arxiv.org/pdf/2303.00900)
2. [`Local contrastive loss with pseudo-label based self-training for weakly-supervised medical image segmentation Paper`](https://arxiv.org/pdf/2112.09645)


## Structure of file system
```
└── bvm_training # the root folder
    ├── data # you can create a folder for storing the dataset (check in the next section)
    ├── models # you can create a folder for storing the pretrained weights
    ├── jobs # the folder contains all the jobs that run on snellius.
    ├── output_jobs # the folder contains the logs of the trainings.
    ├── trans_bvm # files for the training of the original model.
    ├── trans_bvm_self_supervised # files for the training of the weakly-supervised version of the model.
    ├── environment.yml # conda environment.
    ├── make_video.py # creates a video based on the masks and the original frames.
    └── tranmission_map.py # this file creates for all the transmitted images.
```
## Input folder
In the case of SMOKE5K dataset, you can download the zip from the [`link`](https://drive.google.com/file/d/11TM8hsh9R6ZTvLAUzfD6eD051MbOufCi/view). Place the zip in a folder named "data" (or you can use a different name). The unziping of this file can be done in python by the transmission_map.py. After the file is unzipped, the folder structure will be the following:

```
└── data # the root folder of data
    ├── SMOKE5K # dataset folder after unzip
        └── SMOKE5K
            ├── test # test images
                ├── gt # masks
                    ├── img1.png
                    └── ... 
                └── img # RGB images
                    ├── img1.png
                    └── ... 
            └── train # train images
                ├── gt # masks
                    ├── img1.png
                    └── ... 
                ├── trans # transmission masks (only necessary in training)
                    ├── img1.png
                    └── ... 
                └── img # RGB images
                    ├── img1.png
                    └── ... 
    └── ijmond_data # custom dataset
        ├── test # test images
            ├── gt # masks
                ├── img1.png
                └── ... 
            └── img # RGB images
                ├── img1.png
                └── ... 
        └── train # train images
            ├── gt # masks
                ├── img1.png
                └── ... 
            ├── trans # transmission masks (only necessary in training)
                ├── img1.png
                └── ... 
            └── img # RGB images
                ├── img1.png
                └── ..
```
You can add a new custom dataset as shown above, but keep the same folder structure as demonstrated. This mean that all instances of a sample (RGB image, mask, transmission map) should have the same name. Also, you should only use the "gt","trans" and "img" folder names to refer to the masks' folder, transmission maps' folder and RGB images' folder respectively.

## Navigate to the main folder
```
cd bvm_training
```

## Transmission estimation
Before starting training you should creat the transmission maps for each RGB image of the dataset. To do this, use the following command for SMOKE5K dataset:
```
python transmission_map.py --dataset_name "SMOKE5K" --dataset_zip "data/SMOKE5K.zip" --output "data/SMOKE5K" --mode "train"
```
Or for a custom dataset:
```
python transmission_map.py --dataset_name "ijmond" --output "data/ijmond_data" --mode "train"
```
## Original Code
For training and testing the original BVM model in the SMOKE5K dataset use the commands below:
```
python trans_bvm/train.py --dataset_path "data/SMOKE5K/SMOKE5K/train" --save_model_path "models/train_SMOKE5K"
python trans_bvm/test.py
```
For training and testing the original BVM model in a custom dataset with pretrained weights use the commands below:
```
python trans_bvm/train.py --dataset_path "data/ijmond_data/train" --pretrained_weights "models/ucnet_trans3_baseline/Model_50_gen.pth" --save_model_path "models/finetune"
python trans_bvm/test.py
```
For snellius, use:
```
sbatch jobs/train.job
```
When you use the argument "--pretrained_weights" in the train.py file and set a path for a pretrained model, the algorithm will load the pretrained weights and perform fine-tuning. If the argument is set to None, then the training will start from scratch. The argument "--save_model_path" is used to set the path where the weights will be stored during training.

For testing, you should specify the folder where images are strored in the variable "dataset_path" inside the code (line 22) and the model you want to use (line 24).

## Weakly-supervised
For training and testing the weakly-supervised BVM model in the SMOKE5K dataset use the commands below:
```
python trans_bvm/train.py --contrastive_loss_weight 0.1 --labeled_dataset_path "data/SMOKE5K/SMOKE5K/train" --unlabeled_dataset_path "data/ijmond_data/train" --save_model_path "models/ss__no_samples_1000" --aug False --no_samples 1000
python trans_bvm/test.py 
```
Below is the explanation of the arguments:
1. contrastive_loss_weight: the weight for contrastive loss in the total summation of losses.
2. labeled_dataset_path: the path for the dataset with known ground truth.
3. unlabeled_dataset_path: the path for the dataset with pseudo labels (you can create the pseudo labels by running the python test.py in your data).
4. save_model_path: the folder for storing the new weights.
5. aug: if you want to have augmentations in the unlabeled set (because it is small).
6. no_samples: the number of pixels that you want to use in the contrastive loss from each class.

For snellius, use the following command by changing the arguments when necessary:
```
sbatch jobs/train.job
```

## Evaluation
In order to get the evaluation metrics you can use the eval.py from each trans_bvm version (the original and the weakly-supervised). eval_opacity.py is used to calculate the metrics for the high and low opacity smoke seperately.

For the SMOKE5K dataset:
```
python trans_bvm/eval.py --dataset_path "./data/SMOKE5K/SMOKE5K/SMOKE5K/test/img" --gt_path "./data/SMOKE5K/SMOKE5K/SMOKE5K/test/gt" --save_path "./results/" --model_path "./models/finetune/Model_50_gen.pth"

python trans_bvm/eval_opacity.py --dataset_path "./data/SMOKE5K/SMOKE5K/SMOKE5K/test/img" --gt_path "./data/SMOKE5K/SMOKE5K/SMOKE5K/test/gt" --save_path "./results/" --model_path "./models/finetune/Model_50_gen.pth"
```

For the custom dateset:
```
python trans_bvm/eval.py --dataset_path "./data/ijmond_data/test/img" --gt_path "./data/ijmond_data/test/gt" --save_path "./results/" --model_path "./models/finetune/Model_50_gen.pth"

python trans_bvm/eval_opacity.py --dataset_path "./data/ijmond_data/test/img" --gt_path "./data/ijmond_data/test/gt" --save_path "./results/" --model_path "./models/finetune/Model_50_gen.pth"
```
For the weakly-supervised model you can use the same exactly commands.