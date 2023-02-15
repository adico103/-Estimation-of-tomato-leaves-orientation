
import pandas as pd
import feather
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import csv
import os
import params_update

hpc = params_update.hpc

# from select_next_step import select_next 

if  hpc:
  device = torch.cuda.current_device()
if not hpc:
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TomatoLeafDataset(Dataset):

    def __init__(self,data_type,INPUT_SIZE=86, colab=True,files= 9):
      folder = ''
      self.data_type = data_type
      if colab:
          folder = '/content/drive/My Drive/Thesis/experiment_data/joints/'

      if data_type=='combos_3':
        INPUT_SIZE = (INPUT_SIZE-12)
        name = 'combos_3_'+ str(int(INPUT_SIZE/6)) + '_joints.feather'
      
      if data_type=='combos_3_no_pose':
        name = 'combos_3_no_pose_'+ str(int(INPUT_SIZE/6)) + '_joints.feather'

      if data_type=='combos_2_no_pose':
        name = 'combos_2_no_pose_'+ str(int(INPUT_SIZE/4)) + '_joints.feather'
      
      elif data_type=='Regular':
        # INPUT_SIZE = (INPUT_SIZE-12)
        name = 'combos_1_'+ str(int(INPUT_SIZE/2)) + '_joints.feather'
        
      
      elif data_type=='Contour_old':
        INPUT_SIZE = (INPUT_SIZE-6)
        name = 'combos_2_'+ str(int(INPUT_SIZE/4)) + '_joints.feather'
      


      # else:
      #   print('wrong datatype')
      
      data_path = folder+name
      if (data_type=='combos_3' or data_type=='combos_3_no_pose'):
        data = []
        
        for file in range(files):
          print('loading file ',file+1,'/', files)
          new_name = name[:-15]
          new_name = new_name + '_joints'+'_'+ str(file+1)+ '.feather'
          data.append(feather.read_dataframe(new_name)) 

        len_data = data[files-1].iloc[-1]['index']


      else:
        data = feather.read_dataframe(data_path)
        len_data = len(data)


      x=[]
      y=[]
      Index = []
      leaf_number= []
      self.data = data
      subsection = 0
      start_index = 0
      
      for i in range (len_data):
        if ((i%100)==0):
          print('loading data:', i, '/',len_data)
        if (data_type=='Contour_old' or data_type=='Regular' or data_type=='combos_2_no_pose'):
          x_old = data.iloc[i]['input']
          y_old = np.array(np.stack(data.iloc[i]['leaf_normal']).squeeze())
          ind_old = data.iloc[i]['index']
          leaf_num_old = data.iloc[i]['leaf_number']

        elif (data_type=='combos_3' or data_type=='combos_3_no_pose'):
         
          if i> data[subsection].iloc[-1]['index']:
            subsection+=1
            start_index = data[subsection].iloc[0]['index']
          x_old = data[subsection].iloc[i-start_index]['input']
          y_old = data[subsection].iloc[i-start_index]['leaf_normal']
          y_old = np.array(np.stack(y_old)).squeeze()
          ind_old = data[subsection].iloc[i-start_index]['index']
          leaf_num_old = data[subsection].iloc[i-start_index]['leaf_number']


        x.append(x_old)
        y.append(y_old)
        Index.append(ind_old)
        leaf_number.append(leaf_num_old)

      self.x = torch.FloatTensor(x)
      self.y = torch.FloatTensor(y)
      self.Index = Index
      self.n_samples = len(x)
      self.leaf_number = leaf_number

    def __getitem__(self,index):
      return self.x[index] , self.y[index], self.Index[index]
    
    def __len__(self):
      return self.n_samples

  


def merge_images(load,i,j):
    im1 = load['croped_image'][int(i)].reshape(180,180,3)
    im2 = load['croped_image'][int(j)].reshape(180,180,3)
    merged_img = np.vstack((im1,im2))
    return merged_img.flatten()

class TomatoLeafPairsDataset(Dataset):

    def __init__(self,data_type,colab=True):

      if colab: 
        data_path = '/content/drive/My Drive/Thesis/experiment_data/joints/crop.feather'
        indexes_file = '/content/drive/My Drive/Thesis/experiment_data/joints/new_pairs_file.feather'

      else:
        data_path = 'crop.feather'
        indexes_file = 'new_pairs_file.feather'

      self.data_type = data_type
      data = feather.read_dataframe(data_path)
      pairs_ind = feather.read_dataframe(indexes_file)
      ind_1 = []
      ind_2 = []
      y=[]
      Index = []
      input = []

      
      for i in range (len(data)):

        ind_1.append(int(pairs_ind.iloc[i]['index_1']))
        ind_2.append(int(pairs_ind.iloc[i]['index_2']))
        y.append(np.array(np.stack(data.iloc[i]['leaf_normal']).squeeze()))
        Index.append(i)
        input.append(pairs_ind.iloc[i]['input'])
        

      self.ind_1 = ind_1
      self.ind_2 = ind_2
      self.y = torch.FloatTensor(y)
      self.Index = Index
      self.n_samples = len(ind_1)
      self.data = feather.read_dataframe(data_path)
      self.sample_to_image = lambda X:np.transpose(X,(2,0,1))
      self.input = input

    def __getitem__(self,index):
      if self.data_type == 'Pairs':
        merged_img = merge_images(self.data,self.ind_1[index],self.ind_2[index])
        merged_img = np.reshape(merged_img,np.array([360,180,3]))
        merged_img = self.sample_to_image(merged_img)
        merged_img= merged_img-np.min(merged_img)
        merged_img= merged_img/(np.max(merged_img))

        return torch.FloatTensor(merged_img) , self.y[index], self.Index[index]

      if self.data_type == 'Contour':
        # combined_Contour = combineContour(self.data,self.ind_1[index],self.ind_2[index],self.input_size)
        combined_Contour = self.input[index]
        return torch.FloatTensor(combined_Contour) , self.y[index], self.Index[index]
def last_batch_size(indexes,batch_size):    
  batch = len(indexes) - np.floor(len(indexes)/batch_size)*batch_size
  if batch ==1:
    indexes = indexes[:-1]
  return indexes

def find_split_leaves(dataset,percents,batch_size,datatype):
    max_leaf_number = max(dataset.leaf_number)
    train_upper_bound = int(percents[0]*max_leaf_number)
    test_upper_bound = train_upper_bound+ int(percents[1]*max_leaf_number)
    leaf_numbers= np.array(dataset.leaf_number)
    train_indexes = np.array(dataset.Index)[np.where(np.logical_and(leaf_numbers>=0, leaf_numbers<=train_upper_bound))]
    train_indexes = last_batch_size(train_indexes,batch_size)
    test_indexes = np.array(dataset.Index)[np.where(np.logical_and(leaf_numbers>=train_upper_bound+1, leaf_numbers<=test_upper_bound))]
    test_indexes = last_batch_size(test_indexes,batch_size)
    vaild_indexes = np.array(dataset.Index)[np.where(np.logical_and(leaf_numbers>=test_upper_bound+1, leaf_numbers<=max_leaf_number))]
    vaild_indexes = last_batch_size(vaild_indexes,batch_size)
    indexes = [train_indexes,test_indexes,vaild_indexes]

    return indexes #train, test, valid

def split_data (datatype,percents,dataset,total_size,batch_size,device,random = False) : 
    
    print('Data Type: ' ,datatype)
    percent_of_train = percents[0]
    percent_of_test = percents[1]
    if random:
    
      print('total number of samples',total_size)
      train_size = int(total_size*percent_of_train)
      print('number of train samples',train_size)
      test_size = int(total_size*percent_of_test)
      print('number of test  samples',test_size)
      valid_size = total_size-test_size-train_size
      print('number valid of samples',valid_size)
      lengths = [train_size, test_size , valid_size]
      train_data , test_data , valid_data = torch.utils.data.random_split(dataset, lengths)
      
    else:
      split_indexes = find_split_leaves(dataset,[percent_of_train,percent_of_test],batch_size,datatype)
      train_data = torch.utils.data.Subset(dataset,split_indexes[0])
      test_data = torch.utils.data.Subset(dataset,split_indexes[1])
      valid_data = torch.utils.data.Subset(dataset,split_indexes[2])
      total_size = len(train_data)+ len(test_data) + len(valid_data)
      print('total number of samples',total_size)
      print('number of train samples',len(train_data))
      print('number of test  samples',len(test_data))
      print('number valid of samples',len(valid_data))

    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
    train_loader = DataLoader(dataset=train_data, batch_size=int(batch_size) , shuffle=True, **kwargs)
    test_loader = DataLoader(dataset=test_data, batch_size=int(batch_size) , shuffle=True, **kwargs)
    valid_loader = DataLoader(dataset=valid_data, batch_size=int(batch_size) , shuffle=True, **kwargs)

    print ("Data loaded!")
    return train_loader,test_loader,valid_loader
    
    

def load_dataset(input_size, colab=True ,datatype = 'Regular',files = 9):
  if datatype == 'Regular':
      dataset = TomatoLeafDataset(datatype,input_size,colab=colab)
      total_size = dataset.n_samples

  if (datatype == 'Pairs' or datatype == 'Contour' ):
    dataset = TomatoLeafPairsDataset(datatype,colab=colab)
    total_size = dataset.n_samples

  if (datatype == 'Contour_old' or  datatype=='combos_3'):
    dataset = TomatoLeafDataset(datatype,input_size,colab=colab,files=files)
    total_size = dataset.n_samples
  
  if (datatype == 'combos_2_no_pose'):
    dataset = TomatoLeafDataset(datatype,input_size,colab=colab)
    total_size = dataset.n_samples
  
  if (datatype == 'combos_3_no_pose'):
    dataset = TomatoLeafDataset(datatype,input_size,colab=colab)
    total_size = dataset.n_samples

  return dataset,total_size



  


def load_model(input_size,output_size,dropout,n_layers, model_name,colab=True) -> nn.Module:

    """Load the model corresponding to the name given.

    Args:
    model_name: the name of the model, one of: SimpleNet, XceptionBased.

    Returns:
    model: the model initialized, and loaded to device.
    """
    if not colab:
      from models import LinearModel, Xception, SimpleNet, My_model,ResNet, OpenPose

    models = {
    'Conv': SimpleNet(output_size),
    'Xception':Xception(output_size),
    'LinearModel':LinearModel(input_size,output_size,num_stage=n_layers,p_dropout=dropout),
    'OpenPose': OpenPose(output_size),
    'My_model' : My_model(output_size,dropout,n_layers),
    'ResNet' : ResNet(output_size),
    # 'select_next': select_next(input_size,output_size,num_stage=n_layers,p_dropout=dropout,colab = colab)

    }

    if model_name not in models:
        raise ValueError(f"Invalid Model name {model_name}")

    print(f"Building model {model_name}...")
    model = models[model_name]
    model = model.to(device)
    return model



def get_nof_params(model: nn.Module) -> int:
    """Return the number of trainable model parameters.

    Args:
        model: nn.Module.

    Returns:
        The number of model parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_exp_parameters(number,file):
  csvreader = csv.reader(file)
  rows = []
  for row in csvreader:
      rows.append(row)
  model = rows[number][1]
  data_type = rows[number][2]
  return str(model), str(data_type)

def save_comet_params(logging_parameters,exp_name,best_loss):
  params = {'model_name': [], 'dataset_type': [],
                      'optimizer_name': [], 'train_per': [], 'test_per': [],
                      'exp_num': [], 'epochs': []}
  string = "Experiment_"+str(logging_parameters.num_exp)+"_"
  
  try:
    sub_exp = int(exp_name.split(string,1)[1])
  except:
    sub_exp = 0
  params['model_name'] = logging_parameters.model_name
  params['dataset_type'] = logging_parameters.dataset_type
  params['optimizer_name'] = logging_parameters.optimizer_name
  params['train_per'] = logging_parameters.train_per
  params['test_per'] = logging_parameters.test_per
  params['exp_num'] = logging_parameters.num_exp
  params['epochs'] = logging_parameters.n_epochs
  params['sub_exp'] = sub_exp
  params['best_loss'] = best_loss
  params['lr'] = logging_parameters.learning_rate
  params['n_layers'] = logging_parameters.n_layers
  params['batch_size'] = logging_parameters.batch_size
  params['dropout'] = logging_parameters.dropout_per
  params['optimizer'] = logging_parameters.optimizer_type
  params['loss_f'] = logging_parameters.loss_f

  params_data = pd.DataFrame(params, index=[0])
  folder = 'comet_experiments/'
  param_name = folder+exp_name+ '_params'

  #saving params
  params_data.columns = params_data.columns.astype(str)
  params_data.reset_index(drop=True)
  params_data.to_feather(param_name+'.feather')


def exp_not_done(params,colab=False):
  # if the experiment not done yet- return True
  print('Experiment parameters:')
  print(params)
  compare = [params.lr,
        params.train_percent,
        params.test_percent,
        params.batch_size,
        params.epochs,
        params.optimizer,
        params.dropout,
        params.n_layers,
        params.loss_function]
  folder = 'experiments/'
  if colab:
    folder = 'colab_experiments/'
  for filename in os.listdir(folder):
    if filename.startswith("Experiment_"+str(params.num_exp)):
      if filename.endswith("params.feather"):
        load = feather.read_dataframe(folder+filename)
        loss_f = 'angle'
        if load['loss_f'][0]==2:
          loss_f = 'cos'

        this_exp = [load['lr'][0],
        load['train_per'][0],
        load['test_per'][0],
        load['batch_size'][0],
        load['epochs'][0],
        load['optimizer_name'][0],
        load['dropout'][0],
        load['n_layers'][0],
        loss_f]

        if this_exp==compare:
          print('Experiment with the same parameters has already been done,')
          print('moving to next experiment')
          return False

  return True
