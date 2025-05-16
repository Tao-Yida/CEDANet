# Instructions for running the docker of the Baseline

## Structure of file system
```
└── bvm_docker # the root folder
    ├── Dockerfile # the file to build the docker
    └── inference # the folder with all the necessary files 
        ├── model # the folder with the codes of the model
        ├── monitor.py # the file for checking for new images in the given directory 
        ├── process_image.py # the file to inferece an image to the model 
        └── requirements.txt # the requirements for the docker
```

## Build the docker
```
cd bvm_docker # where the Dockerfile is
sudo docker build -t bvm:0.0.4 .
```

## Build the docker
Below is the command to run the docker. You only need to edit the paths:
1. LOCAL_OUTPUT_FOLDER: is the main folder that you want to store the output.
2. LOCAL_MODELS_FOLDER: is the folder where the pretrained weights are stored.
3. LOCAL_INPUT_FOLDER: is the input folder the includes all the images e.g. "/tmpdata/ijmond-camera-timemachine/kooks_fabriek_2/".
4. Model_50_gen.pth: this is the name of the file with the stored weights. You can change it.
```
 sudo docker run --rm -it --gpus all -v "$(pwd)/LOCAL_OUTPUT_FOLDER:/app/outputs" -v "$(pwd)/LOCAL_MODELS_FOLDER:/app/models" -v "$(pwd)/LOCAL_INPUT_FOLDER:/app/data" bvm:0.0.4 --img_folder "data" --output_folder "outputs" --pretrained_weights "models/Model_50_gen.pth"
```

## Input file system
The input to the docker can have the following structure:
```
└── LOCAL_INPUT_FOLDER # the root folder
    ├── 2024-03-03 # the folder with images of the specific date
        ├── 1709593185.jpg # image
        ├── ... # images
        └── 1709593195.jpg # complete directory. No more new images.
    └── 2024-03-04 
        ├── 1709593196.jpg # image
        ├── ... # images
        └── ... # incomplete directory. meaning that new images are currently added
```
The algorithm will process all the folders in chronological order and stay in the last one to wait recursively for new images. When a new folder appears the algorithm attents to the new folder.

## Output file system
The output of the docker has the following structure:
```
└── LOCAL_OUTPUT_FOLDER # the main output folder
    ├── 2024-03-03 # the folder with masks of the specific date
        ├── 1709593185.jpg # mask
        ├── ... # masks
        └── 1709593195.jpg # mask
    └── 2024-03-04 
        ├── 1709593196.jpg # mask
        ├── ... # masks
        └── 1709593195.jpg # mask
```