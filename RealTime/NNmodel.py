import torch
import feather 
import numpy as np

hpc = False

if  hpc:
  device = torch.cuda.current_device()
if not hpc:
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(input_size,output_size,dropout,n_layers) :

    """Load the model corresponding to the name given.

    Args:
    model_name: the name of the model, one of: SimpleNet, XceptionBased.

    Returns:
    model: the model initialized, and loaded to device.
    """
    from models import LinearModel

    models = {
    'LinearModel':LinearModel(input_size,output_size,num_stage=n_layers,p_dropout=dropout),
    # 'select_next': select_next(input_size,output_size,num_stage=n_layers,p_dropout=dropout,colab = colab)

    }
    model_name = 'LinearModel'


    print(f"Building model {model_name}...")
    model = models[model_name]
    model = model.to(device)
    return model


class RealTimeModel:
    def __init__(self,
                INPUT_SIZE = 132,
                OUTPUT_SIZE = 3,
                num_exp = 5,
                sub_exp = 5
                ):
                print('loading trained model')

                self.Name = 'Experiment_'+str(num_exp)+'_'+str(sub_exp)
                self.model_path = 'checkpoint//'+ str(self.Name)+'.pt'
                exp_param_path = 'experiments//'+self.Name+'_params.feather'
                self.parameters = feather.read_dataframe(exp_param_path)
            #       self.model = load_model(self.INPUT_SIZE,OUTPUT_SIZE,self.parameters['dropout'][0],
            # self.parameters['n_layers'][0],self.parameters['model_name'][0],colab=False)

                self.model = load_model(INPUT_SIZE,OUTPUT_SIZE,self.parameters['dropout'][0],
                self.parameters['n_layers'][0])
                self.model.load_state_dict(torch.load(self.model_path,map_location=torch.device('cpu'))['model'])
                self.model.eval()

    def estimate(self,input):
        
        input  = torch.FloatTensor(input).unsqueeze(0)
        with torch.no_grad():
          output =  self.model(input).to(device)
          vec = output.detach().numpy()


        return vec



#         
