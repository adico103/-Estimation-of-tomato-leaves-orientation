import pandas as pd
import params_update

api_key="uaCrdTwo4wqyBYgwZEiyfT0MT"
project_name="Thesis_experiments"
workspace="adico"

config = params_update.config

class my_opt():
  def __init__(self,         
                ):
    self.lr = config["parameters"]["lr"]
    self.n_layers = config["parameters"]["n_layers"]
    self.batch_size = config["parameters"]["batch_size"]
    self.dropout = config["parameters"]["dropout"]
    self.optimizer = config["parameters"]["optimizer"]
    self.loss_f = config["parameters"]["loss_f"]
    self.combinations={'lr': [],'n_layers': [],'batch_size': [],'dropout': [],'optimizer': [],'loss_f': [] }

    for lr in self.lr:
      for n_layers in self.n_layers:
        for batch_size in self.batch_size:
          for dropout in self.dropout:
            for optimizer in self.optimizer:
              for loss_f in self.loss_f:
                self.combinations['lr'].append(lr)
                self.combinations['n_layers'].append(int(n_layers))
                self.combinations['batch_size'].append(int(batch_size))
                self.combinations['dropout'].append(dropout)
                self.combinations['optimizer'].append(optimizer)
                self.combinations['loss_f'].append(loss_f)
                
    self.combinations = pd.DataFrame(self.combinations)

  def get_experiments(self,project_name,workspace):
    experiments = []
    self.project_name = project_name
    self.workspace = workspace
    count = 0
    for i in range (len(self.combinations)):
      count+=1
      combination = self.combinations.iloc[i]
      experiments.append(experiment(combination=combination,sub_exp=count))

    return experiments



class experiment():
  def __init__(self,
                combination,
                sub_exp,
                ):

      self.sub_exp = sub_exp
      self.params = combination
      self.best_loss = []

      self.metrics = {'train_loss': [], 'train_acc': [],
                      'val_loss': [], 'val_acc': [],
                      'test_loss': [], 'test_acc': [], 'lr': []}
      self.logging_parameters = {'model_name': [], 'dataset_type': [],
                      'optimizer_name': [], 'train_per': [], 'test_per': [],
                      'exp_num': [], 'epochs': []}

  def log_parameters(self,logging_parameters):
    self.logging_parameters['model_name'] = logging_parameters.model_name
    self.logging_parameters['dataset_type'] = logging_parameters.dataset_type
    self.logging_parameters['optimizer_name'] = logging_parameters.optimizer_name
    self.logging_parameters['train_per'] = logging_parameters.train_per
    self.logging_parameters['test_per'] = logging_parameters.test_per
    self.logging_parameters['exp_num'] = logging_parameters.num_exp
    self.logging_parameters['epochs'] = logging_parameters.n_epochs
    

  def get_parameter(self,param_name):
    return self.params[param_name]

  def set_name(self, exp_name):
    self.exp_name = exp_name

  def log_metric(self,metric_name,metric_value,epoch=False):
    if epoch:
      self.metrics[metric_name].append(metric_value)
    else:
      self.best_loss = metric_value

  def end(self):
    #saving all data

    self.logging_parameters['sub_exp'] = self.sub_exp
    self.logging_parameters['best_loss'] = self.best_loss
    self.logging_parameters['lr'] = self.params['lr']
    self.logging_parameters['n_layers'] = self.params['n_layers']
    self.logging_parameters['batch_size'] = self.params['batch_size']
    self.logging_parameters['dropout'] = self.params['dropout']
    self.logging_parameters['optimizer'] = self.params['optimizer']
    self.logging_parameters['loss_f'] = self.params['loss_f']

    params_data = pd.DataFrame(self.logging_parameters, index=[0])
    hist_data =  pd.DataFrame(self.metrics)
    folder = 'experiments/'
    param_name = folder+self.exp_name+ '_params'
    hist_name = folder+self.exp_name+ '_hist'

    #saving params
    params_data.columns = params_data.columns.astype(str)
    params_data.reset_index(drop=True)
    params_data.to_feather(param_name+'.feather')

    #saving hist
    hist_data.columns = hist_data.columns.astype(str)
    hist_data.reset_index(drop=True)
    hist_data.to_feather(hist_name+'.feather')






