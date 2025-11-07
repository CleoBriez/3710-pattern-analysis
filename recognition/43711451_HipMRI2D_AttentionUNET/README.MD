# Title
    2D HipMRI Dataset Segmented using Attention U-Net 

# Problem Description
    Segment the HipMRI Study on Prostate Cancer (see Appendix for link) using the processed 2D slices (2D images) with the 2D CAN with all labels having a minimum Dice similarity coefficient of 0.75 on the test set on the prostate label. You will need to load Nifti file format and sample code provided. 

# Algorithm Description
## Data Preparation
    Data was pre-sorted into training, validation and testing datasets as well as segment masks for each.
    The data was wrapped as if it were 3D in order to standardise the inputs as they were multiclass rather than binary. 
    After it was wrapped, testing data was shuffled, normalised and put into a dataset and dataloader for training and validation.

## Model Architecture 
### Filters
    The encoder steps were filtered from [64, 128, 256, 512, 1024], the inverse of [1024, 512, 256, 128, 64] was there for the decoder
### Downsampling (Encoder)
    The encoder carries the input through the double convultions and skip connections through each filter. 

### Upsampling (Decoder)
    The decoder acts inversely though to the encoder and receives the input though the skip connections and the bottleneck. 

### Bottleneck
    The bottleneck applies the double convolutions using the 3x3 kernel.
    The output is then upsampled through the decoder. 

### Attention Gates
    As I've tried to make this an Attention U-Net by using Attention gates to add additional context to the U-Net structure by adding a layer of mapping semantic context from the decoder and spatial detail from the encoder alongside the regular connections. 

## Model Training
    Model was trained using all training samples and validated using the validation samples. 
    A random seed was also applied to try and generalise the training and prevent issues with just learning the training data. 

## Model Performance
    The model was run and saved using batches of size 16 and initially 25 epochs using the full testing and validation sample. 
    I actually got 25 epochs out of it but messed up my save location so it didn't save. I really would've liked to refine that :c.
    I was running on an Nvidia RTX 3060Ti with 8GB of VRAM. 
    After 20~ Epochs the was overfitting and this could have been optimised better by experimenting with smaller subsets with increased epochs.
    
## Model Testing
    After training, the model was tested using the predict function to then be put ont 

# How it Works 
## File Structure 
.
├── recognition
│   ├── mask_output
│   │   ├── pred_mask_case_040_week_0_slice_0.nii.gz
│   │   └── pred_mask_case_040_week_0_slice_0.nii.png
│   │   ├── pred_mask_case_040_week_1_slice_0.nii.gz
│   │   └── pred_mask_case_040_week_1_slice_0.nii.png
│   │   ├── pred_mask_case_040_week_2_slice_0.nii.gz
│   │   └── pred_mask_case_040_week_2_slice_0.nii.png
│   ├── saved_models
│   │   └── full_set_5_epochs.pth
│   ├──dataset.py
│   ├──modules.py
│   ├──predict.py
│   ├──README.MD
│   ├──train.py
│   └── utils.py
├── LICENSE
└── README.md

## Requirements & Dependencies
    Torch (for Keras layers, models, and tensoring)
    NumPy (for numerical operations)
    Matplotlib (for plotting)
    NiBabel (for neuroimaging data handling)
    NiLearn (For image resampling)
    Tqdm (for progress bars)
    Pathlib (for filesystem path manipulations)
    Random (for random seeding)

## Future Improvements
    I honestly would really like to adapt this towards 3D Datasets and visualising it in nicer/funner ways. 
    I also would like to just clean up the hard-coded parameters and make it nicer to just run in a more central way. 
    I would've liked to add functionality to the predict section to compare the created masks to the provided one in a more clear way like analysing the area that was accurate. 
    Honestly a lot of the predict section was really exciting but I didn't really expand on it as much as I would've liked to. 
    Tbh this doc too ;~;

## Usage & Reproduction Steps
    For Training:
        python3 train.py
    For predictions and Visualisations
        python3 predict.py

    Hyperparameters are present at the top of both to allow for customisation to how it's run. 
    Set the dataset locations in the dataset.py path variable too. 
    Subsets are able to be configured using the SUBSET param to make it smaller. (0 = all the data)

# Visualisations
    Output prediction visualisations are present in the 'recognition\43711451_HipMRI2D_AttentionUNET\mask_output' folder
