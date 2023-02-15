exp_num = 3  # choose between 3,5,7,8,9
exp_folder = 'E:\Adi\Leaf Orientation Estimation\\\data_exp\\'
exp_data = exp_folder+'data\\'
Total_data = 'Total_data.feather'
Total_filltered_data = 'Total_Filltered_data.feather'
Total_filltered_pairs = 'Total_Filltered_pairs.feather'
trash_directory = exp_folder+'Cropped\\trash\\'
norms_and_joints_folder = 'data_joints_and_normals\\'
joints_folder = 'data_with_joints\\'
raw_data_directory = 'raw_data\\' 
hpc = False

config = {

    "parameters": {
      "lr": [0.05,0.5,1],
      "n_layers" :[2,4,5],
      "batch_size" :[32,64],
      "dropout":  [0,0.3],
      "optimizer":[1],
     
       },
    
    }
