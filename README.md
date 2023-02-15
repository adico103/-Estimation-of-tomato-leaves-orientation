# Estimation of Tomato Leaves Orientation for Early Diseases Detection Using Deep Neural Network Model

# Introduction
## Main Steps
The Project contains 5 main steps:
1. Experiment - data aquisition
2. Data Pre-Processing
3. Deep learning model training
4. Analasys and results
5. Real time pipeline

## Notes
1. clone repository
2. all data is in the hard drive in the lab , inside `Adi\Tomato_leaves` the data folders (that are not in this repository) are:
- `data_exp` - all of the images that were collected in the data aquistion process
- `code\experiments` - saved parameters for the best models evaluated. 
- `code\checkpoint` -  saved best models. 
- `RealTime\experiments` - saved parameters for the best model evaluated. 
- `RealTime\checkpoint` -  saved best model.




3. crate a folders tree in your own folder as described:
```bash
├───Analayse_data
│   ├───conf
│   ├───cont
│   └───route
├───checkpoint
├───data_joints_and_normals
├───data_with_joints
├───experiments
├───Image Collector
│   ├───.vscode
│   ├───daq
│   │   ├───rawdata
│   │   └───__pycache__
│   ├───gui
│   │   ├───.vscode
│   │   ├───img
│   │   └───__pycache__
│   └───__pycache__
├───out
├───raw_data
├───RealTime
│   ├───checkpoint
│   ├───data
│   │   └───norm_imgs
│   ├───experiments

```

4. in `params_update.py` update the follwoing parameters:


| Parameter | Description |
| --- | --- |
| `exp_folder` | change to the folder where your experiment data is saved   |
| `hpc` | change to `True` if you are running on hpc |
| `exp_num` | change according to the experiment number you want to run (further explanation in the "Deep learning model training" section) |
| `config` | change the hyper parameters you want to examine (further explanation in the "Deep learning model training" section) |


# Experiment - data aquisition
To perform the data aquisiton process, make sure you are connected to the turntable,UR5 robotic arm and realsense camera.
Modify the line of code in `turntable.py` file that is in `daq` folder to update the port of the turn-table arduino communication

```
self.ser = serial.Serial('COM4', baudrate=9600, timeout=1)
```
 Modify the line of code in `ur5.py` file that is in `daq` folder to update the HOST port for the UR5 robotic arm communication

```
HOST = "192.168.1.113" # The remote host
```

Open `Image Collector` directory and Run `main.py` - Credit: Yarden Akaby
This window will apeare:

<p align="center"> 
<img src="https://github.com/Arl2023/Estimation-of-tomato-leaves-orientation/blob/main/gui.PNG" width=50%>
</p>


First press on UR5, Turn and Camera to make sure there is a connection with all of them
Then press on Test- this will start a program that rotates the turntable, changes the elevation angle of the robotic arm and saves the images in the experiment data folder. The files are saved as feather files under names as this example: "N14_d0.3_p0.0_r0_O0.feather" - This is an image of leaf No. 14, the distance from the robotic arm to the leaf is 0.3 m, the elevation angle of the robot is 0 degrees, the number of rotation of the turntable is 0, and the primary orientation on the mechanism is 0 (as described in my thesis). example for a saved image:

<p align="center"> 
<img src="https://github.com/Arl2023/Estimation-of-tomato-leaves-orientation/blob/main/N14_d0.3_p0.0_r8_O0new.png" width=50%>
</p>



# Data Pre-Processing
After collecting the labeled images in the data aquisition process. we will prepare the different datasets by processing the raw data
Run `data_pre_process_main.py`. This script perform all of the steps to prepare the data to the model. Notice that this script is combined with a few steps. each step take a long time, and saves files that will be used for the next one. so in case the process stops - you can continue form where you stopped. The different steps are:

| Step | Description |
| --- | --- |
| create_raw_data | saves a feather file for every leaf in `raw_data` folder. the file combines all of the different images files into one   |
| extract_contour_joints | image processing stage - as described in my thesis. saves feather files for every leaf in folder `data_with_joints` that contains the experiment parameters (without the images themselves) and an extra column of "joints" that represents the pixels on the contour of the leaf  |
| compute_norms | Calculating the normal vector by the initial image of every leaf (as described in my thesis), saving the same dataframe as last step with additional column "leaf_normal" in folder `data_joints_and_normals` |
| combine_all | creats `Total_data.feather` file: combines all of the feather files in to one file with additional "index_raw_data" column that states the matching index in the raw data file of each leaf (in folder `raw_data`) |
| make_total_filltered_file | creats `Total_filtered_data.feather` file: filteres the bad samples out from the `Total_data.feather` according to images that are moved to the folder `\data_exp\Cropped\trash` |
| make_1_image_data | creates a simplified file of the input (20 equally spaced pixels) and the output (the leaf normal) for every image |
| make_2_imgs_data | creates a databse of 2 images (as described in my thesis) of pairs of images of the same leaf from different elevation angles of the robot |
| make_3_imgs_data | creates a databse of 3 images (as described in my thesis) of combinations of images of the same leaf from different elevation angles of the robot|

After running all of the steps, the files that will be used to train the model are: 
1. `combos_1_20_joints.feather`
2. `combos_2_20_joints.feather`
3. `combos_3_20_joints_#.feather` (when # goes from 1 to 10 different files)

# Deep learning model training
## Parameters

### Different models
There are 5 types of models to run (same architecture, the database is different) change the `exp_num` in the `params_update.py` file to run the scpecific model type you desire. (detailed explanation about the datasets difference is in section 3.5.2 in the thesis file). Notice- *RTD = Robotic Transform Data

| exp_num | Description |
| --- | --- |
| 3 | 2 images model + RTD  |
| 5 | 3 images model + RTD |
| 7 | 1 image model |
| 8 | 2 images model |
| 9 | 3 images model |

### Optimization
In the training process, all of the combinations of the hyperparameters that were chosen are evaluated. Change `config` in `params_update.py` to add/change the parameters you want to examine:

| Hyperparameter | Description |
| --- | --- |
| `lr` | learning rate (float) |
| `n_layers` | number of hidden layers (int) |
| `batch_size` | batch size (int) |
| `dropout`| dropout percent (float, 0 to 1)|
| `optimizer` | "1" for Adam , "2" for SGD  |

## Run
Run `main.py` - this script trains a deep learning model (as described in my thesis) to estimate the leaf normal vector. This is done by evaluationg a few conbinations of hyperparameters for every dataset selected (defined as sub_exp). Don't worry - if the process stops for some reason, it will continue from the last combination the next time you run the program.

### Saved data
In the running process, a few files are saved:

| File | Folder | Description |
| --- | --- | --- |
| `Experiment_#_@_hist.feather` | experiments | contains the history of the train, validation and test loss and acccuracy in every epoch |
| `Experiment_#_@_params.feather` | experiments |contains the experiment parameters for every sub_exp: number of sub_exp, best model loss, and the hyperparameters  |
| `Experiment_#_@.pt` | checkpoint | saved torch model |
| `Experiment_#_@.json`| out| contains the history of the train, validation and test loss and acccuracy in every epoch |

* "#" is exp_num, "@" is sub_exp


# Analasys and results
Run `analyze_main.py` - this script contains a few steps:

## Getting All Experiments Information

in this code section, 2 types of files are saved:
1. experiment info:
every sample in the train,validation and test sets, is saved with its own parameters: 
* model params regarding this sample: input, output (of the model), targets (labeld leaf normal), raw_data_indexes (every sample is related to its own raw_data file)
* experiment varaibles of the sample: leaf_number, orientation, rotation, elevation, distance
* calculated varaibels for datasets that contain 2 or 3 images: : delta_elevation (the change between the elevation angles of the robot between the images in the sample), delta_distance (the change between the distances of the robot from the leaf between the images in the sample), route_name (short representation of the route the robotic arm performed in this sample), route_index (index equivalent to the route_name).
2. experiment routs info:
* a file containing a list of all of the unique routs the robotic arm has performed in all of the samples.

The files are saved for every exp and sub_exp in `Analayse_data` folder.

| File  | File Type |
| --- | --- |
| `Experiment_#_@_train.feather`| experiment info - for train data|
| `Experiment_#_@_valid.feather`| experiment info - for validation data|
| `Experiment_#_@_test.feather`| experiment info - for test data|
| `Experiment_#_@_routs.feather`| experiment routs info|

* "#" is exp_num, "@" is sub_exp

## Saving Final Results

this script goes over all of the experiments and save a csv file with all of the final results: 
for each experiment+sub_experiment it saves the experiment varaibles (learing rate, number of layers, batch size and dropout) as well as the train,validation and test avrage score (error) of the model estimation. The file is saved as `final_table.csv` in `Analayse_data` folder.


## Plots

### compare example

the script plots an example (if `conf` folder) of comparing the best models (for 1,2,3 images with RTD) to view the normal prediction of each of the models. for example:

<p align="left">
<img src="https://github.com/Arl2023/Estimation-of-tomato-leaves-orientation/blob/main/True%20vs%20Predicted%20normal_error_13.5.png" width=50%>
</p>

### Cumulative propability graphs

the script plots cumulative propability graphs for the best models for 2 images model+RTD and 3 images model+RDT. to see what percent of the test samples reached under a certain error, under a certain route of the robot elevation angles. the plots are saved in `route` folder. Notice that there are many possible routs so there are a few graphs for each model (each contains 6 routs, and are numberd by the "sucseess" of these routes) for example the 6 best routs for the "3 images+RTD" model:
<p align="center">
<img src="https://github.com/Arl2023/Estimation-of-tomato-leaves-orientation/blob/main/Error_angle_percents_as_function_of_delta_elevation_1_Experiment_5_5.png.svg" width=50%>
</p>

### Analyze additional images contribution
the script plots graphs to evaluate the contribution of images to existing ones, as it compares "1 imgae model", "2 images model+RTD" and "3 images model+RTD". the graphs are saved in `cont` folder (detailed explenation about the graphs is in the thesis file). example of the plot:

<p align="center"> 
<img src="https://github.com/Arl2023/Estimation-of-tomato-leaves-orientation/blob/main/cont13_impr_per%3D85.26.png" width=50%>
</p>


# Real time pipeline
Now for the fun part - making everything work together! - The pipeline is working on the best model selected, working by taking 3 images in an optimal route and saving the equivalent robotic pose, and evaluating the leaf normal vector.
## Preperations
1. Before you get started, make sure you have the best model saved in `RealTime\checkpoint` folder, as well as its parameters in `RealTime\experiments` (see Notes at the begining of this readme file)
2. Connect the UR5 and the realsense camera placed on the robotic arm. make sure the connections is working well.
3. In `MainRealTime.py` change `realtime` to True.
4. In `MainRealTime.py` change `current_measure` to True -  if you want the robot to reach the leaf with the gripper.

## Run
place the leaf in front of the robotic arm (0.5 m) and run `MainRealTime.py`, the UR5 will take 3 images of the leaf in different elevation angles, and will save the parameters of the files in `RealTime\data` folder, as well as the RGB and the green filltered image, for example:

<p align="center">
<img src="https://github.com/Arl2023/Estimation-of-tomato-leaves-orientation/blob/main/realtime1.png" width=50%>
</p>

Next, the model will evaluate the leaf normal and save an image in `RealTime\data\norm_imgs` folder:

<p align="center">
<img src="https://github.com/Arl2023/Estimation-of-tomato-leaves-orientation/blob/main/realtime2.png" width=50%>
</p>

Then the script will print the Leaf Normal Vector (in the UR5 base coordinate system), and the desired robotic end-effector pose to grasp the leaf. For Example:

```
Leaf Normal Vector: [ 0.46 -0.85  0.27]
Desired Robotic Pose: [-0.22       -0.10673258  0.0787869   0.32484781 -1.48965856 -0.27070651]

```

 If you changed `current_measure` to True, the robotic arm will reach to grasp the leaf.



