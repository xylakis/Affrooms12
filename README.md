# Affrooms12: Affect in Spatial Navigation, a study of Rooms

The repository contains all the necessary files to run the Leave-One-Out Cross-Validation Random Forest classifier for this project. Make sure that
all the necessary packages are installed according to the 'requirements.txt' file.

=== Affect in Spatial Navigation: A study of Rooms. ===

Contributors: Emmanouil Xylakis, Antonios Liapis, Georgios N Yannakakis 

Description: This study proposes a framework to identify and model the effects that our perceived environment can have by taking into consideration illumination 
and structural form while acknowledging its temporal dimension. To study this, we recruited 100 participants via a crowd-sourcing platform in order to
annotate their perceived arousal or pleasure shifts while watching videos depicting spatial navigation in first person view. Participants’ annotations 
were recorded as time-continuous unbounded traces, allowing us to extract ordinal labels about how their arousal or pleasure fluctuated as the camera moved between 
different rooms. 

Published: IEEE Transactions on Affective Computing

DOI Bookmark: 10.1109/TAFFC.2024.3493761

# Github Repository: 

### Scripts
The `scripts` that are included in the Github repository are the main Python scripts for running the Random Forest classifier (RF). 
Two files are included:
- `Random_Forest_Leave_One_Out.py`: Contains the RF architecture for hyperparameter tuning and train-test on the processed data files.
- `Random_Forest_defs.py`: Contains the necessary functions for tuning the RF classifier.

### Data

#### Ordinal labels
The `Ordinal labels` directory contains all the necessary data to run the `` script. It includes 2 main sub-folders, one for 
each affect label (Arousal-Pleasure). Within each affect label directory a 'Train_Test' directory and a 'Validation' directory contain the corresponding
data files for each Leave-One-Out Cross-Validation Fold. Targets and Inputs for the Random Forest Classifier can be found in seperate files. 

Data files contain the following: 
- Each affect dimension uses 3 affect signal measures (Mean, Amplitude & Gradient). 
- Each inputs file includes 13 feature columns. 
- Each targets file contains a single column with binary values. 

#### Processed affect traces
The `Processed affect traces` directory contains two data files (one for each affect label), containing all the aquired continuous traces from all participants 
of this study. All processed data have been through a MinMax [0,1] Normalization and Sampled at 250ms intervals. The columns include: Timestamps, Affect 
annotation value, Run id, Video id, Room id and Participant id. 

OSF Storage: 

### Wiki Images
The `Wiki_images` folder includes all images used for the repository's wiki pages.

### Affroooms_12_videofiles 
The 'Affroooms_12_videofiles' directory contains all 54 pre-recorded videos within the Unreal Engine, depicting all the walkthroughts that were used as stimuli
as part of the affect annotation task. 
