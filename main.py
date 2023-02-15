
import torch
from torch import nn, optim
from dataclasses import dataclass
import os
from my_opt import my_opt
import params_update

hpc = params_update.hpc
colab = False #is the model runs on colab

if not colab:
  from trainer import LoggingParameters, Trainer
  from utils import load_dataset, load_model,get_exp_parameters, split_data, exp_not_done

if  hpc:
  device = torch.cuda.current_device()
if not hpc:
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if colab:
  OUTPUT_DIR = '/content/drive/My Drive/Thesis/out'
  CHECKPOINT_DIR = '/content/drive/My Drive/Thesis/checkpoint'
  exp_file_path = '/content/drive/My Drive/Thesis/experiments.csv'

else:
  OUTPUT_DIR = 'out'
  CHECKPOINT_DIR = 'checkpoint'
  exp_file_path = 'experiments.csv'

OUTPUT_SIZE = 3


#for_opt
lr=0.01
batch_size=32
dropout=0.0
n_layers=2

#const
train_percent = 0.8
test_percent = 0.1
epochs = 100
n_milstones = 3

def get_milstones(epochs, n):
  milestones = []
  section_len = int(epochs/(n+1))
  for i in range (n):
    milestones.append(section_len*(i+1))
  return milestones

milestones = get_milstones(epochs, n_milstones)

@dataclass
class Args:
  """Data class holding paraeters for logging"""
  lr: float
  train_percent: float
  test_percent: float
  batch_size: int
  epochs: int
  model: str
  optimizer: str
  data_type: str
  dropout: float
  n_layers: int
  loss_function: str
  num_exp : int

project_name="Thesis_experiments"
workspace="adico"

config = {
    # We pick the Bayes algorithm:
    "algorithm": "bayes",

    # Declare your hyperparameters in the Vizier-inspired format:
    "parameters": {
      "lr": {"type": "discrete", "values": [0.05,0.5,1,5,10,15,20]},
      "n_layers":{"type": "integer", "min": 1, "max": 6},
        "batch_size":{"type": "discrete", "values": [32,64,128,256]},
        "dropout": {"type": "discrete", "values": [0,0.3,0.5,0.7]},
       "optimizer": {"type": "discrete", "values": [1, 2]},
      "loss_f": {"type": "discrete", "values": [1,2],
        
       },
        
       },

    # Declare what we will be optimizing, and how:
    "spec": {"metric": "best_loss","objective": "minimize",
    }
    }

opt = my_opt()



def main(args,experiment,exp_name):
    
    percents = args.train_percent,args.test_percent
    model_name = args.model
    model = load_model(INPUT_SIZE,OUTPUT_SIZE,args.dropout,int(args.n_layers),model_name,colab=colab)
    train_loader,test_loader,valid_loader = split_data(datatype=args.data_type,percents=percents,dataset=dataset,total_size=total_size,batch_size=args.batch_size,device=device,random = False)
    criterion = nn.CosineSimilarity()
    cos_f = nn.CosineSimilarity()
    

    # Build optimizer
    optimizers = {
        'SGD': lambda: optim.SGD(model.parameters(),
                                 lr=args.lr),
                                #  momentum=args.momentum),
        'Adam': lambda: optim.Adam(model.parameters(), lr=args.lr),
    }

    optimizer_name = args.optimizer
    if optimizer_name not in optimizers:
        raise ValueError(f'Invalid Optimizer name: {optimizer_name}')

    print(f"Building optimizer {optimizer_name}...")
    optimizer = optimizers[args.optimizer]()
    print(optimizer)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    optimizer_params = optimizer.param_groups[0].copy()
    # remove the parameter values from the optimizer parameters for a cleaner
    # log
    del optimizer_params['params']

    # Batch size
    batch_size = args.batch_size

    # Training Logging Parameters
    logging_parameters = LoggingParameters(model_name = model_name,
                                           dataset_type = args.data_type,
                                           optimizer_name = optimizer_name,
                                           optimizer_params = optimizer_params,
                                           optimizer_type = args.optimizer,
                                           learning_rate = args.lr,
                                           dropout_per = args.dropout,
                                           n_layers = args.n_layers,
                                           batch_size = args.batch_size,
                                           train_per = args.train_percent,
                                           test_per = args.test_percent,
                                           n_epochs = args.epochs,
                                           num_exp=args.num_exp,
                                           loss_f=args.loss_function
                                           )


    # Create an abstract trainer to train the model with the data and parameters
    # above:
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      scheduler = scheduler,
                      criterion=criterion,
                      cos_f=cos_f,
                      batch_size=batch_size,
                      train_dataset=train_loader,
                      validation_dataset=valid_loader,
                      test_dataset=test_loader,
                      dropout=args.dropout,
                      loss_f=args.loss_function,
                      num_exp = args.num_exp,
                      data_type = args.data_type,
                      )
                      
    # Train, evaluate and test the model:
    best_valid_loss = trainer.run(experiment, epochs=args.epochs,CHECKPOINT_DIR = CHECKPOINT_DIR, exp_name=exp_name,logging_parameters=logging_parameters)
    

    return best_valid_loss

if __name__ == '__main__':
  num_exp = params_update.exp_num # change only here
  combos = 2
  if num_exp == 7:
     combos = 1
  if num_exp == 5:
     combos = 3
  n_joints = 20
  
  INPUT_SIZE = n_joints*2*combos+6*(combos-1)

  with open(exp_file_path, 'r',encoding= 'unicode_escape') as file:
    model, data_type = get_exp_parameters(num_exp,file)
    file.close()
  print('loading dataset:',data_type)
  dataset,total_size = load_dataset(INPUT_SIZE,colab=colab,datatype=data_type)
  
  for experiment in opt.get_experiments(
    project_name=project_name,
    workspace=workspace):
    #get exp params
    opt_num = experiment.get_parameter("optimizer")
    loss_f_num = experiment.get_parameter("loss_f")
    lr = experiment.get_parameter("lr")
    batch_size=experiment.get_parameter("batch_size")
    dropout=experiment.get_parameter("dropout")
    n_layers=experiment.get_parameter("n_layers")
    if opt_num==1:
        optimizer_name = 'Adam'
    if opt_num==2:
      optimizer_name = 'SGD'
    # if loss_f_num==1:
    #   loss_f_name = 'angle'
    # if loss_f_num==2:
    #   loss_f_name = 'cos'      
    loss_f_name = 'cos' 
    # experiment parameters
    args = Args(lr=lr,
                  train_percent=train_percent,
                  test_percent=test_percent,
                  batch_size=batch_size,
                  epochs=epochs,
                  model=model,
                  optimizer=optimizer_name,
                  data_type=data_type,
                  dropout=dropout,
                  n_layers=n_layers,
                  loss_function=loss_f_name,
                  num_exp=num_exp)
    
    if exp_not_done(args,colab=colab):

      exp_name =  'Experiment_'+ str(num_exp)
      output_filename = exp_name +  '.json'
      output_filepath = os.path.join(OUTPUT_DIR, output_filename)
      sub_exp = 0
      if os.path.exists(output_filepath):
          models_exist = os.listdir(OUTPUT_DIR)
          matching = [s for s in models_exist if exp_name in s]
          sub_exp = str(len(matching)+1)
          exp_name = exp_name + '_'+ sub_exp
          output_filename = exp_name + '.json'
          output_filepath = os.path.join(OUTPUT_DIR, output_filename)

      experiment.set_name(exp_name)
      best_valid_loss = main(args,experiment,exp_name)

      experiment.log_metric("best_loss", best_valid_loss)

      experiment.end()

    

