from utils import load_dataset, load_model, split_data
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
import feather
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import os
import random
import re
from analyze_help_functions import draw_2_normals_on_image ,draw_4_normals_on_image
import math 
from itertools import combinations
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics

hpc=False
colab = False
OUTPUT_SIZE = 3
n_joints = 20

analayze_folder = 'Analayse_data//'
conf_folder = 'Analayse_data/conf/'
image_cont = 'Analayse_data/image_addition_contribution.feather'
case_studie_test = 'Analayse_data/case_studie_test.feather'
case_studie_train = 'Analayse_data/case_studie_train.feather'
case_studie_valid = 'Analayse_data/case_studie_valid.feather'
decision_test_data = 'Analayse_data/decision_test_data.feather'
decision_train_data = 'Analayse_data/decision_train_data.feather'
decision_valid_data = 'Analayse_data/decision_valid_data.feather'
desicion_paths = [decision_train_data,decision_valid_data,decision_test_data]



red = (255 ,0 ,0)
blue = (0 ,0 ,255)

other = (100 ,100 ,100)
green_2 = (0 ,100 ,0)
colors = [red,blue,other,green_2] 

if  hpc:
  device = torch.cuda.current_device()
if not hpc:
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ExpMaster:
    def __init__(self,
                num_exp,
                ):
        self.num_exp = num_exp

    def find_sub_exp(self, load = False):
        folder = 'experiments//'
        end = '_params.feather'
        if load:
            folder = analayze_folder
            end = '_test.feather'
        sub_exps = []
        for filename in os.listdir(folder):
            start = 'Experiment_'+str(self.num_exp)+'_'
            
            if filename.startswith(start):
                if filename.endswith(end):
                    pattern = str(start)+"(.*?)"+str(end)
                    try:
                        sub_exp = re.search(pattern, filename).group(1)
                        
                    except: sub_exp =''
                    sub_exps.append(sub_exp)
        
        self.sub_exps = sub_exps

    def load_all_data(self):
        ## data loading:
        print('loading raw data files')
        # self.pairs_indexes = feather.read_dataframe('Pairs_indexes.feather')
        self.Total_Filltered_data = feather.read_dataframe('Total_Filltered_data.feather')
        
        if self.num_exp == 7:
            # self.data_in_network = feather.read_dataframe('combos_2_20_joints.feather')
            self.datatype = 'Regular'
            self.data_combos = feather.read_dataframe('combos_1.feather')
            combos = 1
            self.INPUT_SIZE = n_joints*2*combos+6*(combos-1)

        if self.num_exp == 3:
            # self.data_in_network = feather.read_dataframe('combos_2_20_joints.feather')
            self.datatype = 'Contour_old'
            self.data_combos = feather.read_dataframe('combos_2.feather')
            combos = 2
            self.INPUT_SIZE = n_joints*2*combos+6*(combos-1)

        if self.num_exp == 5:
            combos = 3
            self.datatype = 'combos_3'
            self.INPUT_SIZE = n_joints*2*combos+6*(combos-1)
            # self.data_in_network = feather.read_dataframe('combos_3_20_joints.feather')
            data = []
            files= 9
            name = 'combos_3_'
            for file in range(files):
                print('loading file ',file+1,'/', files)

                new_name = name + str(file+1)+ '.feather'
                temp_data = feather.read_dataframe(new_name)
                temp_data = temp_data[['index','index_1', 'index_2','index_3']]
                data.append(temp_data) 

            self.data_combos = pd.concat(data)
        
        if self.num_exp == 8:
            # self.data_in_network = feather.read_dataframe('combos_2_20_joints.feather')
            self.datatype = 'combos_2_no_pose'
            self.data_combos = feather.read_dataframe('combos_2.feather')
            combos = 2
            self.INPUT_SIZE = n_joints*2*combos

        if self.num_exp == 9:
            combos = 3
            self.datatype = 'combos_3_no_pose'
            self.INPUT_SIZE = n_joints*2*combos
            # self.data_in_network = feather.read_dataframe('combos_3_20_joints.feather')
            data = []
            files= 9
            name = 'combos_3_'
            for file in range(files):
                print('loading file ',file+1,'/', files)

                new_name = name + str(file+1)+ '.feather'
                temp_data = feather.read_dataframe(new_name)
                temp_data = temp_data[['index','index_1', 'index_2','index_3']]
                data.append(temp_data) 

            self.data_combos = pd.concat(data)

        self.data_set,self.total_size = load_dataset(self.INPUT_SIZE,colab=colab,datatype=self.datatype)
                    
    def save_exps_info(self,sub = [], all_sub=True):

        self.load_all_data()
        if all_sub:
            self.find_sub_exp()
        else:
            self.sub_exps = sub

        for sub_exp in self.sub_exps:
            new_exp = ExpAnalyser(self.num_exp,
                                    sub_exp,
                                    self.INPUT_SIZE,                                     
                                    )

            new_exp.data_combos  = self.data_combos
            new_exp.data_set = self.data_set
            new_exp.total_size = self.total_size
            new_exp.Total_Filltered_data = self.Total_Filltered_data
            new_exp.get_all_exp_info()
            new_exp.save_all_data()    

    def load_exps_info(self,load=True):

        self.find_sub_exp(load=load)
        exps_table = []
        for sub_exp in self.sub_exps:
            new_exp = ExpAnalyser(self.num_exp,
                                    sub_exp,  
                                    load= not load                                                       
                                    )
            # new_exp.plot_hist(save=True)
            table = new_exp.get_table()            
            exps_table.append(table)
            # new_exp.plot_vectors_examples('train',10)
        
        return pd.concat(exps_table)
        # return exps_table

class ExpAnalyser:
    def __init__(self,
                num_exp,
                sub_exp,
                INPUT_SIZE=40,
                load = False
                ):

        self.num_exp = num_exp
        if self.num_exp==3:
            self.num_img = 2
        if self.num_exp==5:
            self.num_img = 3
        if self.num_exp==7:
            self.num_img = 1
        if self.num_exp==8:
            self.num_img = 2
        if self.num_exp==9:
            self.num_img = 3
        self.INPUT_SIZE = INPUT_SIZE
        self.sub_exp = sub_exp
        self.Name = 'Experiment_'+str(num_exp)+'_'+str(sub_exp)
        if sub_exp=='':
            self.Name = 'Experiment_'+str(num_exp)

        def creat_folder(path):
            isExist = os.path.exists(path)
            if not isExist:
            # Create a new directory because it does not exist 
                os.makedirs(path)

        self.Total_Filltered_data = feather.read_dataframe('Total_Filltered_data.feather')
        print('Experiment Name: ', self.Name )
        self.hist_graphs = 'Analayse_data//hist/'
        self.routs_graphs = 'Analayse_data//route/'
        self.exp_compare_graphs = 'Analayse_data//exp_compare/'
        self.vectors_compare = 'Analayse_data//vectors_compare/'
        creat_folder(self.hist_graphs)
        creat_folder(self.routs_graphs)
        creat_folder(self.exp_compare_graphs)
        creat_folder(self.vectors_compare)
        exp_param_path = 'experiments//'+self.Name+'_params.feather'
        self.parameters = feather.read_dataframe(exp_param_path)

        if not load:
            self.model_path = 'checkpoint//'+ str(self.Name)+'.pt'
            ## exp parameters loading:
            print('loading experiment parameters')
            
     
        if load:
            train_data_path = 'Analayse_data//'+self.Name+'_train.feather'
            test_data_path = 'Analayse_data//'+self.Name+'_test.feather'
            valid_data_path = 'Analayse_data//'+self.Name+'_valid.feather'
            route_data_path = 'Analayse_data//'+self.Name+'_routs.feather'
            hist_path = 'experiments//'+self.Name+'_hist.feather'
            self.train_data = feather.read_dataframe(train_data_path)
            self.test_data = feather.read_dataframe(test_data_path)
            self.valid_data = feather.read_dataframe(valid_data_path)
            self.hist = feather.read_dataframe(hist_path)
            if self.num_exp!=7:
                self.route_dict = feather.read_dataframe(route_data_path)

    def get_all_exp_info(self):


        ## network data loading:
        print('loading network train, validation and test data')


        self.train_loader,self.test_loader,self.valid_loader = split_data(datatype=self.parameters['dataset_type'][0],
                                                                            percents=[self.parameters['train_per'][0],self.parameters['test_per'][0]],
                                                                            dataset=self.data_set,
                                                                            total_size=self.total_size,
                                                                            batch_size=self.parameters['batch_size'][0],
                                                                            device=device,
                                                                            random = False)
  

        
        ## loading the model
        print('loading trained model')
        self.model = load_model(self.INPUT_SIZE,OUTPUT_SIZE,self.parameters['dropout'][0],
            self.parameters['n_layers'][0],self.parameters['model_name'][0],colab=False)
        
        if not hpc:
            self.model.load_state_dict(torch.load(self.model_path,map_location=torch.device('cpu'))['model'])
        else:
            self.model.load_state_dict(torch.load(self.model_path)['model'])

        self.model.eval()

        def calc_angles(outputs, targets):
            criterion = nn.CosineSimilarity()
            loss = (1-torch.abs(criterion(targets,outputs)))
            angle = torch.arccos(1-loss)*180/(np.pi)
            return angle
            
        def get_outputs(self,loader):

            if loader=='test':
                loader = self.test_loader
            if loader=='train':
                loader = self.train_loader
            if loader=='valid':
                loader = self.valid_loader

            total_inputs = []
            total_targets = []
            total_indexes = []
            total_outputs = []
            total_angles = []

            for batch_idx, (inputs, targets ,index) in enumerate(loader):
                with torch.no_grad():
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = self.model(inputs).to(device)
                    angles = calc_angles(outputs, targets)
                if batch_idx==0:
                    shape_inputs = inputs.shape[-1]
                    shape_targets = targets.shape[-1]
                    shape_outputs = outputs.shape[-1]

                total_inputs = np.append(total_inputs,inputs.cpu())
                total_targets = np.append(total_targets,targets.cpu())
                total_indexes = np.append(total_indexes,index.cpu())
                total_outputs = np.append(total_outputs,outputs.cpu())
                total_angles = np.append(total_angles,angles.cpu())

            total_inputs = total_inputs.reshape(int(len(total_inputs)/shape_inputs),shape_inputs)
            total_targets = total_targets.reshape(int(len(total_targets)/shape_targets),shape_targets)
            total_outputs = total_outputs.reshape(int(len(total_outputs)/shape_outputs),shape_outputs)

            data = {'inputs':[],'targets':[],'indexes':[],'outputs': [],'angles': []}
            data = pd.DataFrame(data)
            data.columns = data.columns.astype(str)
            data['inputs'] = (total_inputs.tolist())
            data['targets'] = (total_targets.tolist())
            data['indexes'] = (total_indexes.tolist())
            data['outputs'] = (total_outputs.tolist())
            data['angles'] = (total_angles.tolist())
            
            return data

        ## calculating model outputs

        print('calculating outputs for train data')
        self.train_data = get_outputs(self,'train')
        print('calculating outputs for validation data')
        self.valid_data = get_outputs(self,'valid')
        print('calculating outputs for test data')
        self.test_data = get_outputs(self,'test')

        def get_row_data_index(dataset):
            if dataset=='test':
                dataset = self.test_data
            elif dataset=='train':
                dataset = self.train_data
            elif dataset=='valid':
                dataset = self.valid_data

            row_data_indexes_list = []
            for index in range (len(dataset)):
                row_data_indexes = []
                new_index = int(dataset['indexes'][index])
                if self.num_exp!=7:
                    row_data_indexes.append(self.data_combos.iloc[new_index]['index_1'])
                    row_data_indexes.append(self.data_combos.iloc[new_index]['index_2'])
                if self.num_exp==5:
                    row_data_indexes.append(self.data_combos.iloc[new_index]['index_3'])
                if self.num_exp==9:
                    row_data_indexes.append(self.data_combos.iloc[new_index]['index_3'])
                if self.num_exp==7:
                    row_data_indexes.append(self.data_combos.iloc[new_index]['index'])


                row_data_indexes_list.append(row_data_indexes)

            dataset['raw_data_indexes'] = row_data_indexes_list

            return dataset
        
        # match row data indexes to the outputs 

        print('matching indexes of row data to train dataset...')
        self.train_data = get_row_data_index('train')
        print('matching indexes of row data to validation dataset...')
        self.valid_data = get_row_data_index('valid')
        print('matching indexes of row data to test dataset...')
        self.test_data = get_row_data_index('test')

        def match_data(dataset):
            if dataset=='test':
                dataset = self.test_data
            elif dataset=='train':
                dataset = self.train_data
            elif dataset=='valid':
                dataset = self.valid_data

            leaf_number = []
            # croped_image = []
            orientation = []
            rotation = []
            elevation = []
            distance = []

            for index in range (len(dataset)):
                    
                raw_data_indexes = dataset['raw_data_indexes'][index]
                if self.num_exp!=7:
                    leaf_number.append(np.array(self.Total_Filltered_data['leaf_number'][raw_data_indexes])[0])
                    orientation.append(np.array(self.Total_Filltered_data['orientation'][raw_data_indexes])[0])
                    rotation.append(np.array(self.Total_Filltered_data['rotation'][raw_data_indexes])[0])
                    elevation.append(np.array(self.Total_Filltered_data['elevation'][raw_data_indexes]))
                    distance.append(np.round(np.array(self.Total_Filltered_data['distance'][raw_data_indexes]),1))
                    
                    check_1 = np.array(self.Total_Filltered_data['orientation'][raw_data_indexes])[1]-np.array(self.Total_Filltered_data['orientation'][raw_data_indexes])[0]
                    check_2 = np.array(self.Total_Filltered_data['rotation'][raw_data_indexes])[1]-np.array(self.Total_Filltered_data['rotation'][raw_data_indexes])[0]
                    check_3 = np.array(self.Total_Filltered_data['leaf_number'][raw_data_indexes])[1]-np.array(self.Total_Filltered_data['leaf_number'][raw_data_indexes])[0]

                    if check_1!=0:
                        print(str(check_1),raw_data_indexes)
                    if check_2!=0:
                        print(str(check_2),raw_data_indexes)
                    if check_3!=0:
                        print(str(check_3),raw_data_indexes)

                if self.num_exp==7:
                    leaf_number.append(np.array(self.Total_Filltered_data['leaf_number'][raw_data_indexes])[0])
                    orientation.append(np.array(self.Total_Filltered_data['orientation'][raw_data_indexes])[0])
                    rotation.append(np.array(self.Total_Filltered_data['rotation'][raw_data_indexes])[0])
                    elevation.append(np.array(self.Total_Filltered_data['elevation'][raw_data_indexes])[0])
                    distance.append(np.round(np.array(self.Total_Filltered_data['distance'][raw_data_indexes])[0],1))


            dataset['leaf_number'] = leaf_number
            # dataset['croped_image'] = croped_image
            dataset['orientation'] = orientation
            dataset['rotation'] = rotation
            dataset['elevation'] = elevation
            dataset['distance'] = distance
            
            return dataset

        # match row data info to the outputs 

        print('matching equivalent raw data to train dataset... ')
        self.train_data = match_data('train')
        print('matching equivalent raw data to validation dataset... ')
        self.valid_data = match_data('valid')
        print('matching equivalent raw data to test dataset... ')
        self.test_data = match_data('test')

        def calc_route(dataset):
            if dataset=='test':
                dataset = self.test_data
            elif dataset=='train':
                dataset = self.train_data
            elif dataset=='valid':
                dataset = self.valid_data
            
            delta_elevation = []
            delta_distance = []
            route_name = []



            for index in range (len(dataset)):
                delta_el_temp = []
                delta_dis_temp = []
                elevation = dataset['elevation'][index]
                distance = dataset['distance'][index]
                name = ''
                for i in range (len(elevation)-1):
                    e = np.round((elevation[i+1]-elevation[i]),1)
                    d = np.round((distance[i+1]-distance[i]),1)
                    delta_el_temp.append(e)
                    delta_dis_temp.append(d)
                    name = name + 'e'+str(e)+'d'+str(d)
                
                delta_elevation.append(delta_el_temp)
                delta_distance.append(delta_dis_temp)
                route_name.append(name)

            
            dataset['delta_elevation'] = delta_elevation
            dataset['delta_distance'] = delta_distance
            dataset['route_name'] = route_name

            return dataset
        
        if self.num_exp!=7:
        
            print('Classifying different routes to train dataset... ')
            self.train_data = calc_route('train')
            print('Classifying different routes to valid dataset... ')
            self.valid_data = calc_route('valid')
            print('Classifying different routes to test dataset... ')
            self.test_data = calc_route('test')

            def routs_dictionary():
                all_routs = np.array(self.train_data['route_name'])
                all_routs = np.append(all_routs,np.array(self.valid_data['route_name']))
                all_routs = np.append(all_routs,np.array(self.test_data['route_name']))
                self.unique_routs = np.unique(all_routs)

            print('creating routs dictionary')
            routs_dictionary()

            def index_routs(dataset):
                if dataset=='test':
                    dataset = self.test_data
                elif dataset=='train':
                    dataset = self.train_data
                elif dataset=='valid':
                    dataset = self.valid_data

                route_indexes = []
                for index in range (len(dataset)):
                    route_index = np.where(self.unique_routs==dataset['route_name'][index])
                    route_indexes.append(route_index[0][0])       
                dataset['route_index'] = route_indexes
                return dataset

            print('indexing routs for train dataset... ')
            self.train_data = index_routs('train')
            print('indexing routs for validation dataset... ')
            self.valid_data = index_routs('valid')
            print('indexing routs for test dataset... ')
            self.test_data = index_routs('test')

        def cal_alpha_beta(dataset):
            if dataset=='test':
                dataset = self.test_data
            elif dataset=='train':
                dataset = self.train_data
            elif dataset=='valid':
                dataset = self.valid_data

            alpha = []
            cos_beta = []

            for index in range (len(dataset)):
                normal = dataset['targets'][index]
                cos_beta_temp = np.cos(np.arcsin(normal[2]))
                cos_beta.append(cos_beta_temp)
                alpha_temp = np.arcsin(normal[1]/cos_beta_temp)
                alpha.append(alpha_temp) 

            dataset['alpha'] = alpha
            dataset['cos_beta'] = cos_beta
            return dataset

        print('calculating normal angles for train dataset... ')
        self.train_data = cal_alpha_beta('train')
        print('calculating normal angles for validation dataset... ')
        self.valid_data = cal_alpha_beta('valid')
        print('calculating normal angles for test dataset... ')
        self.test_data = cal_alpha_beta('test')
    
    def save_data_frame(self,name):
        if name=='test':
            dataset = self.test_data
        elif name=='train':
            dataset = self.train_data
        elif name=='valid':
            dataset = self.valid_data
        elif name=='routs':
            dataset = self.unique_routs

        data_frame = pd.DataFrame(dataset)
        data_frame.columns = data_frame.columns.astype(str)
        data_frame.reset_index(drop=True)
        new_name = 'Analayse_data/'+ str(self.Name)+'_'+str(name)+'.feather'
        data_frame.to_feather(new_name)
        print('saving file:',new_name)

    def save_all_data(self):
        self.save_data_frame('test')
        self.save_data_frame('train')
        self.save_data_frame('valid')
        if self.num_exp!=7:
            self.save_data_frame('routs')

    def get_table(self):
        table = self.parameters[["lr", "n_layers","batch_size","dropout"]]
        table['train_score'] = np.round(np.mean(self.train_data['angles']),2)
        table['validation_score'] = np.round(np.mean(self.valid_data['angles']),2)
        table['test_score'] = np.round(np.mean(self.test_data['angles']),2)
        table['Experiment'] = self.num_exp
        table['Sub_exp'] = self.sub_exp

        # # table['image_num'] = str(image_num)
        return table

    def get_img(self,leaf_number,real_raw_indexes):
        filename = 'raw_data/rawdata_'+str(int(leaf_number))+'.feather'     
        load = feather.read_dataframe(filename)
        try:
            shape = np.array(load.iloc[real_raw_indexes]['shape_color'])[0]
            img = load.iloc[real_raw_indexes]['img_color']
        except:
            shape = np.array(load.iloc[int(real_raw_indexes)]['shape_color'])
            img = load.iloc[int(real_raw_indexes)]['img_color']
        try:
            imgs = []
            for im in img:
                imgs.append(im.reshape(shape))
        except:
            imgs = img.reshape(shape)

        return imgs

    def find_matching_img(self,dataset,index):
        if dataset=='test':
            dataset = self.test_data
        elif dataset=='train':
            dataset = self.train_data
        elif dataset=='valid':
            dataset = self.valid_data
        raw_data_indexes = dataset.iloc[index]['raw_data_indexes']
        real_raw_indexes = self.Total_Filltered_data.iloc[raw_data_indexes]['index_raw_data']
        leaf_number = dataset.iloc[index]['leaf_number']
        img = self.get_img(leaf_number,real_raw_indexes)
        return img

    def get_good_eval_index(self,dataset,error):
        if dataset=='test':
            dataset = self.test_data
        elif dataset=='train':
            dataset = self.train_data
        elif dataset=='valid':
            dataset = self.valid_data
        result_index = dataset['angles'].sub(error).abs().idxmin()
        raw_indexes = dataset['raw_data_indexes'][result_index]
        true = dataset['targets'][result_index]
        pred = dataset['outputs'][result_index]
        angle = dataset['angles'][result_index]
        ind = raw_indexes[-1]
        leaf_number = dataset['leaf_number'][result_index]
        real_raw_indexes = self.Total_Filltered_data.iloc[ind]['index_raw_data']
        img = self.get_img(leaf_number,real_raw_indexes)

        return raw_indexes,true,img,pred,angle

    def get_pred_vec(self,dataset,index): 
        if dataset=='test':
            dataset = self.test_data
        elif dataset=='train':
            dataset = self.train_data
        elif dataset=='valid':
            dataset = self.valid_data
        if len(index)==2:
            for i in range(len(dataset)):
                if dataset['raw_data_indexes'][i][0]==index[0]:
                    if dataset['raw_data_indexes'][i][1]==index[1]:
                        # print('yes')
                        ind = i
        else:
            ind= dataset['raw_data_indexes'][dataset['raw_data_indexes']==index[0]].index[0]


        pred = dataset['outputs'][ind]
        angle = dataset['angles'][ind]
        return pred,angle

    def plot_pred_vs_true_3d(self,pred,true,angle,save=True):
        origin = [0,0,0]
        X, Y ,Z = zip(origin)
        U, V ,W = zip(pred)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(X,Y,Z,U,V,W,arrow_length_ratio=0.2, color=['r'],label='predicted')
        X, Y ,Z = zip(origin)
        U, V ,W = zip(true)
        ax.quiver(X,Y,Z,U,V,W,arrow_length_ratio=0.2, color=['b'],label='true')
        plt.legend()
        plt.xlabel("X")
        plt.ylabel("Y")
        ax.set_zlabel('Z')
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)
        
        title = 'True vs Predicted normal 3D\n error angle = '+str(np.round(angle,2))+u'\N{DEGREE SIGN}'
        title_2 = 'True_vs_Predicted_3D'
        name = self.vectors_compare + title_2 +'_' +self.Name+'_error_'+ str(np.round(angle,2))+'.png'
        plt.legend()
        plt.title(title)
        if save:
            plt.savefig(name + '.png')
        else:
            plt.show()

    def plot_pred_vs_true_img(self,pred,true,img,angle,save=True):

        new_img = draw_2_normals_on_image(pred,true,img)
        title = 'True vs Predicted normal'
        name = self.vectors_compare + title +'_' +self.Name+'_error_angle_'+ str(np.round(angle,2))+'.png'
        plt.title(title)
        if save:
            plt.imsave(name,new_img)
        else:
            plt.imshow(new_img)

    def get_random_expamles(self,dataset,n):

        if dataset=='test':
                dataset = self.test_data
        elif dataset=='train':
            dataset = self.train_data
        elif dataset=='valid':
            dataset = self.valid_data

        indexes = random.sample(range(len(dataset)), n)
        angles = dataset['angles'][indexes]
        return indexes,angles
    
    def find_spread_examples(self,dataset,n):
        if dataset=='test':
            dataset = self.test_data
        elif dataset=='train':
            dataset = self.train_data
        elif dataset=='valid':
            dataset = self.valid_data
        eq_indexes = np.linspace(0,len(dataset)-1,n).astype(int)
        indexes = dataset['angles'].argsort()[eq_indexes]
        angles = dataset['angles'][indexes]
        return indexes,angles

    def find_vector(self,dataset,index):
        if dataset=='test':
            dataset = self.test_data
        elif dataset=='train':
            dataset = self.train_data
        elif dataset=='valid':
            dataset = self.valid_data
        pred = dataset.iloc[index]['outputs']
        true = dataset.iloc[index]['targets']
        return pred,true

    def get_joints(self,dataset,index):
        if dataset=='test':
            dataset = self.test_data
        elif dataset=='train':
            dataset = self.train_data
        elif dataset=='valid':
            dataset = self.valid_data

        
        joints=[]
        for i in range(self.num_img):
            start = i*2*n_joints
            jump= 2*n_joints
            joints.append(dataset.iloc[index]['inputs'][start:start+jump])

        return joints

    def plot_vectors_examples(self,dataset,n,best=True,img=True,d3=True):

        if best:
            indexes,angles = self.find_spread_examples(dataset,n)
        else : 
            indexes,angles = self.get_random_expamles(dataset,n)

        

        for index,angle in zip(indexes,angles):
            pred,true = self.find_vector(dataset,index)
            if img:
                imgs = self.find_matching_img(dataset,index)
                joints = self.get_joints(dataset,index)
                self.plot_pred_vs_true_img(pred,true,imgs[-1],angle)
            if d3:
                self.plot_pred_vs_true_3d(pred,true,angle)

    def plot_hist(self, save = False):
        plt.figure()
        epochs = np.arange(1,len(self.hist['train_acc'])+1,1)
        plt.plot(epochs, np.array(self.hist['train_acc']), label = "train")
        plt.plot(epochs, np.array(self.hist['val_acc']), label = "validation")
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss angle' u'\N{DEGREE SIGN}')
        plt.grid()
        title = 'Loss vs Epoch'
        name = 'Loss_vs_Epoch_'
        folder = self.hist_graphs
        path = folder + name+ '_'+str(self.Name[11:]) +'.png'
        if save:
            plt.title(title)
            plt.savefig(path)
        else:
            plt.show()

    def find_tuple_in_columns(self,dataframe_column,vec):
        ind= []
        if self.num_img>2:
            for i in range(len(dataframe_column)):
                if dataframe_column[i][0]==vec[0]:
                    if dataframe_column[i][1]==vec[1]:
                        ind.append(i)
        else:
            ind = dataframe_column[dataframe_column==vec[0]].index
        return ind

    def arange_by_best_elev(self,unique,param,dataset,x_val):
        scores = []
        angles_lists = []
        xs = []
        ys = []
        points = []
        
        for index in range(len(unique)):
            group = unique[index]
            inds = self.find_tuple_in_columns(abs(dataset[param]),group)
            angles_list = np.array(dataset['angles'][inds])
            angles_lists.append(angles_list)
            size = len(angles_list)
            x = np.linspace(0,90,10000)
            y=[]
            for x_angle in x:
                y.append((len(np.where(angles_list<x_angle)[0]))*100/size)
            result_index = (np.abs(x - x_val)).argmin()
            point = x[result_index],y[result_index]
            scores.append(y[result_index])
            xs.append(x)
            ys.append(y)
            points.append(point)
        sorted_ind = np.array(scores).argsort()[::-1]
        unique= unique[sorted_ind]
        
        angles_lists=np.array(angles_lists)[sorted_ind]
        xs = np.array(xs)[sorted_ind]
        ys = np.array(ys)[sorted_ind]
        points = np.array(points)[sorted_ind]

        return unique,angles_lists,xs,ys,points
    
    def plot_error_less_then_1_image(self,dataset,save=True,show=False):
        if dataset=='test':
            dataset = self.test_data
        elif dataset=='train':
            dataset = self.train_data
        elif dataset=='valid':
            dataset = self.valid_data
        colors = ['b','g','r','c','m','y']
        x_val= 15
        
        angles_list = np.array(dataset['angles'])
        x = np.linspace(0,100,10000)
        y=[]
        size = len(angles_list)
        for x_angle in x:
            y.append((len(np.where(angles_list<x_angle)[0]))*100/size)
        result_index = (np.abs(x - x_val)).argmin()
        point = x[result_index],y[result_index]
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        plt.plot(x,y,colors[0] )
        plt.plot( [point[0],0], [point[1],point[1]],colors[-1]+'--', )
        plt.plot([x_val,x_val], [point[1],0],'k--')


        major_ticks = np.arange(0, 101, 20)
        minor_ticks = np.arange(0, 101, 5)
        
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        
        fontsize = 12
        fontsize_2 = 16

        plt.xlabel('Error Angle'+ '\xb0',fontsize = fontsize_2)
        plt.ylabel('Percante with lower Error [%]',fontsize = fontsize_2)

        ax.grid(which='major',axis='both', alpha=1)
        ax.grid(which='minor',axis='both', alpha=0.5)

   
        more_y = np.array([0,point[1]])
        plt.xticks(np.array([0,15,30,60,80,100]))
        # plt.yticks(list(plt.yticks()[0]) + list([more_y][0]),fontsize=fontsize)
        
        
        # plt.xticks()
        name = 'Error_angle_percents'
        title = 'Model '+str(self.num_img)
        path = self.routs_graphs+ name+ '_'+str(self.Name) +'.png'
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 110])

        if save:
            plt.title(title,fontsize=fontsize_2)
            plt.savefig(path+'.svg',format='svg',bbox_inches='tight')
        if show:
            plt.show()

    def plot_error_less_then(self,dataset,param,n_lines=6,save=True,show=False):
        plt.figure()
        x_val= 15
        if dataset=='test':
            dataset = self.test_data
        elif dataset=='train':
            dataset = self.train_data
        elif dataset=='valid':
            dataset = self.valid_data
        if self.num_img>2:
            flat_list = [item for sublist in np.array(abs(dataset[param])) for item in sublist]
            arr= np.array(flat_list).reshape(-1,2)
            unique_values = np.unique(arr,axis=0)

        else:
            unique_values = np.unique(np.array(abs(dataset[param])))
        
        

        unique_values,angles_lists,xs,ys,points = self.arange_by_best_elev(unique_values,param,dataset,x_val)
        arr = np.arange(len(unique_values))
        arr_1 = arr[:-(len(arr)%n_lines)]
        arr_2 = arr[-(len(arr)%n_lines):]
        

        try:
            chunks = np.array_split(arr_1,int(len(arr_1)/n_lines))
            chunks.append(arr_2)

        except:
            chunks = np.array_split(arr_2,1)

        if (len(arr)%n_lines)==0:
            chunks =  np.array_split(arr,int(len(arr)/n_lines))

        plot_num = 0
        for chunk_indexes in chunks:
            chunk_points = []
            plot_num+=1
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            colors = ['b','g','r','c','m','y']
            color_index = 0
            for index in chunk_indexes:
                
                group = unique_values[index]
                # angles_list = angles_lists[index]
                x = xs[index]
                y = ys[index]
                point = points[index]
                chunk_points.append(point)
                if param=='delta_elevation':
                    g=group[0]
                    if self.num_img>2:
                        g=group
                    label = str(g) + u'\N{DEGREE SIGN}'
                if param=='delta_distance': 
                    label = 'Distance change: '+ str(group) +' [m]'
                plt.plot(x,y,colors[color_index], label = label )
                plt.plot( [point[0],0], [point[1],point[1]],colors[color_index]+'--', )
                color_index+=1

            plt.plot([x_val,x_val], [max(np.array(chunk_points)[:,1]),0],'k--')
        
        # plt.plot(x,y, '.')
            plt.legend(title ='Robotic Arm Elevation Angle\n Between images' )
            major_ticks = np.arange(0, 101, 20)
            minor_ticks = np.arange(0, 101, 5)
            
            ax.set_xticks(major_ticks)
            ax.set_xticks(minor_ticks, minor=True)
            ax.set_yticks(major_ticks)
            ax.set_yticks(minor_ticks, minor=True)
            
            fontsize = 12
            fontsize_2 = 16

            plt.xlabel('Error Angle'+ '\xb0',fontsize = fontsize_2)
            plt.ylabel('Percante with lower Error [%]',fontsize = fontsize_2)

            ax.grid(which='major',axis='both', alpha=1)
            ax.grid(which='minor',axis='both', alpha=0.5)

            extraticks_x = x_val
            # extraticks_x = np.linspace(0,40,5)
            y_ins = np.array(chunk_points)[:,1].argsort()
            extraticks_y = np.array(chunk_points)[:,1][y_ins][[0,-1]]
            if ((self.num_exp==5) or (self.num_exp==9)):
                extraticks_y = np.array(chunk_points)[:,1][y_ins][[-1]]
           
            # plt.xticks(list(plt.xticks()[0]) + list([extraticks_x]),fontsize=fontsize)
            plt.xticks(np.arange(0, 45, step=5))
            plt.yticks(list([extraticks_y][0]))
            more_y = np.array([0,20,30,40,50,100])
            plt.yticks(list(plt.yticks()[0]) + list([more_y][0]),fontsize=fontsize)
            
            # plt.xticks()
            name = 'Error_angle_percents_as_function_of_'+str(param)
            title = 'Model '+str(self.num_img)
            # title = 'Percent of data with lower error then Error Angle\n Model '+str(self.num_img)
            # title = 'Error Angle Percents as Function of The Change of Elevation Angle\n Model '+str(self.num_img)
            path = self.routs_graphs+ name+ '_'+str(plot_num)+'_'+str(self.Name) +'.png'
            ax.set_xlim([0, 40])
            ax.set_ylim([0, 110])

            if save:
                plt.title(title,fontsize=fontsize_2)
                plt.savefig(path+'.svg',format='svg',bbox_inches='tight')
            if show:
                plt.show()

        return unique_values

    def plot_parameters(self,dataset,param_1,param_2,save=False):
        plt.figure()
        if dataset=='test':
            dataset = self.test_data
        elif dataset=='train':
            dataset = self.train_data
        elif dataset=='valid':
            dataset = self.valid_data
        x = np.array(dataset[param_1])
        y = np.array(dataset[param_2])
        plt.plot(x,y, '.')
        plt.xlabel(param_1)
        plt.ylabel(param_2)
        name = param_2 + ' as function of '+ param_1
        path = 'graphs//'+ name+ '_'+ str(self.Name)+ '.png'
        if save:
            plt.title(name)
            plt.savefig(path)
        plt.show()

    def arch_accuracy(self,dataset,alpha,cos_beta,threshold_cos_betha,threshold_alpha):
        threshold_alpha = [alpha-threshold_alpha,alpha+threshold_alpha]
        threshold_cos_betha = [cos_beta-threshold_cos_betha,cos_beta+threshold_cos_betha]

        if dataset=='test':
            dataset = self.test_data
        elif dataset=='train':
            dataset = self.train_data
        elif dataset=='valid':
            dataset = self.valid_data

        new_dataset = dataset.iloc[np.where(dataset['alpha'].between(threshold_alpha[0], threshold_alpha[1], inclusive=True))]
        new_dataset = new_dataset.iloc[np.where(new_dataset['cos_beta'].between(threshold_cos_betha[0], threshold_cos_betha[1], inclusive=True))]
        score = np.mean(new_dataset['angles'])
        if new_dataset.empty:
            score = 0
        return score
        
    def prepare_directional_data(self,dataset):
        size_dev = 100
        threshold_cos_betha = 0.1
        threshold_alpha = 0.35

        X = np.linspace(-1,1,size_dev)
        Y = np.linspace(-1,1,size_dev)
        Z = np.zeros((size_dev,size_dev))

        for ind_x in range (len(X)):
            for ind_y in range (len(Y)):
                
                alpha = np.arctan(Y[ind_y]/X[ind_x])
                cos_beta = Y[ind_y]/np.sin(alpha)
                
                Z[ind_x][ind_y] = self.arch_accuracy(dataset,alpha,cos_beta,threshold_cos_betha,threshold_alpha)


        return X,Y,Z

    def plot_normal_direction_analasys(self,dataset,save=False):

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        X,Y,Z = self.prepare_directional_data(dataset)
        X, Y = np.meshgrid(X, Y)

        surf = ax.plot_surface(X, Y, np.array(Z), cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

        # Customize the z axis.
        # ax.set_zlim(-1.01, 1.01)
        ax.zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        ax.zaxis.set_major_formatter('{x:.02f}')

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)


        name = 'Affect of normal direction on results'
        path = 'graphs//'+ name+ '_'+ str(self.Name)+ '.png'
        if save:
            plt.title(name)
            plt.savefig(path)
        plt.show()

    def find_best_route(self,dataset,first_index):
       scores = dataset['angles'][dataset['first_indexes']==first_index]
       min_score = np.min(scores)
       min = scores[scores==min_score]
       min_ind = min.index
       best_route_index = dataset.iloc[min_ind]['route_index']
       deltas = dataset.iloc[min_ind]['delta_elevation']
       distances = dataset.iloc[min_ind]['delta_distance']

       return best_route_index,min_score,deltas,distances

    def plot_points(self,p1,p2,p3,p4,x,y):
        plt.plot(x,y)
        plt.plot([p1[0],p2[0]],[p1[1],p2[1]])
        plt.plot([p3[0],p4[0]],[p3[1],p4[1]])

    def get_geometry(self,joints):
        joints = joints[0:2*n_joints]
        x = joints[0::2]
        y = joints[1::2]
        center = np.mean(x),np.mean(y)
        max_dis = 0
        point_1_save = x[0],y[0]
        point_2_save = x[0],y[0]
        for i in range(len(x)):
            point_1 = x[i],y[i]
            for j in range(len(x)):
                point_2 = x[j],y[j]
                dis = math.dist(point_1, point_2)
                if dis>max_dis:
                    max_dis=dis
                    point_1_save = point_1
                    point_2_save = point_2
        if point_1_save[0]-point_2_save[0]==0:
            slope = 1e10000
        else:
            slope = (point_1_save[1]-point_2_save[1])/(point_1_save[0]-point_2_save[0])

        if np.round(abs(slope),2)==0:
            slope_2 = 1e10
        else:
            slope_2 = -1/slope
        b = center[1]-slope_2*center[0]
        p1 = np.array(center)
        new_x=100
        p2 = [new_x,slope_2*new_x+b]
        # print(slope_2,np.round(slope,2))
        distances = []
        points = []
        for i in range(len(x)):
            p3 = x[i],y[i]
            d = np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
            distances.append(d)
            points.append(p3)
        
        best_points = np.array(points)[np.array(distances).argsort()][:4]
        new_max_dis = 0
        for i in range(len(best_points)):
            point_1 = best_points[i]
            for j in range(len(best_points)):
                point_2 = best_points[j]
                dis = math.dist(point_1, point_2)
                if dis>new_max_dis:
                    new_max_dis = dis
                    new_point_1_save = point_1
                    new_point_2_save = point_2
        if new_point_1_save[0]-new_point_2_save[0]==0:
            new_slope = 1e10
        else:
            new_slope = (new_point_1_save[1]-new_point_2_save[1])/(new_point_1_save[0]-new_point_2_save[0])
        # self.plot_points(new_point_1_save,new_point_2_save,point_1_save,point_2_save,x,y)
        return np.min(x),np.min(y),np.max(x),np.max(y),max_dis,slope,new_max_dis,new_slope

    def plot_route_analazyz(self,best_scores,best_deltas,best_ratios,save=True):
        plt.figure()
        plt.scatter(best_deltas, best_ratios, c=best_scores)


        title = 'Route analysis - Best elevation for specific leaf ratio'
        title_2 = 'Route_analysis_Best_elevation'
        name = self.routs_graphs + title_2 +'_' +self.Name+'.png'
        plt.legend()
        plt.title(title)
        if save:
            plt.savefig(name + '.png')
        else:
            plt.show()
        
    def get_leaf_geo(self,dataset,first_index):
        get_temp_ind =dataset[dataset['first_indexes']==first_index]['inputs'].index[0]
        joints=dataset[dataset['first_indexes']==first_index]['inputs'][get_temp_ind]
        min_x,min_y,max_x,max_y,long_dis,long_slope,short_dis,short_slope = self.get_geometry(joints)  
        del_x = max_x-min_x
        del_y = max_y-min_y
        x_y_ratio = del_y/del_x
        main_ratio = long_dis/short_dis
        return x_y_ratio,main_ratio,long_slope

    def best_routs(self,dataset):
        if dataset=='test':
            dataset = self.test_data
        elif dataset=='train':
            dataset = self.train_data
        elif dataset=='valid':
            dataset = self.valid_data
        dict = self.route_dict 
        lst = np.array(dataset['raw_data_indexes'])
        first_indexes = [item[0] for item in lst]    
        dataset['first_indexes'] = first_indexes
        unique_indexes = dataset['first_indexes'].unique()
        best_route=[]
        scores = []
        x_y_ratios = []
        main_ratios = []
        long_slopes = []
        deltas = []
        distances = []
        for first_index in unique_indexes:
            best_route_index,score,delta,distance = self.find_best_route(dataset,first_index)
            x_y_ratio,main_ratio,long_slope = self.get_leaf_geo(dataset,first_index)
            best_route.append(best_route_index)
            scores.append(score)
            deltas.append(delta.iloc[0][0])
            distances.append(distance.iloc[0][0])
            x_y_ratios.append(x_y_ratio)
            main_ratios.append(main_ratio)
            long_slopes.append(long_slope)
        
        all_data = {'scores': [], 'deltas': [], 'distances': [], 'x_y_ratios': [], 'main_ratios': [], 'long_slopes':[]  }
        all_data['scores'] = scores
        all_data['deltas'] = deltas
        all_data['distances'] = distances
        all_data['x_y_ratios'] = x_y_ratios
        all_data['main_ratios'] = main_ratios
        all_data['long_slopes'] = long_slopes
        name = analayze_folder+'best_route_analysys_'+self.Name+'.csv'
        all_data = pd.DataFrame(all_data)
        all_data.columns = all_data.columns.astype(str)
        save_to_csv(all_data,name)
        best_indexes = np.where(np.array(scores)<15)
        best_scores = np.array(scores)[best_indexes]
        best_deltas = np.array(deltas)[best_indexes]
        best_ratios = np.array(main_ratios)[best_indexes]
        self.plot_route_analazyz(best_scores,best_deltas,best_ratios)
        
def get_best_exps(path,exps):
    sub_exps = []
    for exp in exps:
        table = pd.read_csv(path)
        min_ind = table['test_score'][table['Experiment']==exp].idxmin()
        try:
            sub_exp = int(table['Sub_exp'][min_ind])
        except:
            sub_exp = ''
        sub_exps.append(sub_exp)
    return sub_exps

def plot_vectors_3d(preds,true,angles,error,show=False):
    origin = [0,0,0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y ,Z = zip(origin)
    U, V ,W = zip(true)
    ax.quiver(X,Y,Z,U,V,W,arrow_length_ratio=0.05, color=['g'],label='Ground Truth')

    X, Y ,Z = zip(origin)
    U, V ,W = zip(preds[0])
    ax.quiver(X,Y,Z,U,V,W,arrow_length_ratio=0.05, color=['r'],label='3 images - '+str(np.round(angles[0],2))+u'\N{DEGREE SIGN}')

    X, Y ,Z = zip(origin)
    U, V ,W = zip(preds[1])
    ax.quiver(X,Y,Z,U,V,W,arrow_length_ratio=0.05, color=['b'],label='2 images - '+str(np.round(angles[1],2))+u'\N{DEGREE SIGN}')

    X, Y ,Z = zip(origin)
    U, V ,W = zip(preds[2])
    ax.quiver(X,Y,Z,U,V,W,arrow_length_ratio=0.05, color=['gray'],label='1 images - '+str(np.round(angles[2],2))+u'\N{DEGREE SIGN}')

    
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    ax.set_zlabel('Z')
    ax.set_xlim3d(0, -1)
    ax.set_ylim3d(-0.4, 0.5)
    ax.set_zlim3d(0, 0.5)
    # ax.invert_xaxis()
    # ax.invert_yaxis()
    # 
    
    title = 'True vs Predicted normal for different models'
    title_2 = 'True_vs_Predicted_3D_'+str(error)
    name = conf_folder + title_2 +'.png'
    plt.legend()
    plt.title(title)
    plt.savefig(name + '.svg',format='svg')
    # if show:
    #     plt.show()
    #     stop=9

def plot_vectors_on_image(preds,true,img,error):
    new_img = draw_4_normals_on_image(preds,true,img,colors)
    title = 'True vs Predicted normal'
    name = conf_folder + title + '_error_'+str(error)+'.png'
    plt.title(title)
    plt.imsave(name,new_img)

def fix_vecs(pred,true):
    unit_vector_1 = pred / np.linalg.norm(pred)
    unit_vector_2 = true / np.linalg.norm(true)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)*180/np.pi
    if (180-abs(angle))<angle:
        return -pred
    return pred

def compare_best_models(path,error,exps,show=False):
    best = get_best_exps(path,exps)
    preds=[]
    true = []
    angles = []
    for i in range (len(exps)):
        exp = exps[i]
        sub = best[i]
        new_exp = ExpAnalyser(exp,sub,load=True)
        if i==0:
            index,true,img,pred,angle = new_exp.get_good_eval_index('test',error)
        else:
            index = index[1:]
            pred,angle = new_exp.get_pred_vec('test',index) 
        pred = fix_vecs(pred,true)
        preds.append(pred)
        angles.append(angle)

    plot_vectors_on_image(preds,true,img,error)
    plot_vectors_3d(preds,true,angles,error,show=show)

def save_to_csv(dataframe,table_path):
    dataframe.to_csv(table_path)

def save_final_table(experiments,table_path,old_exp=True):
    final_table=[]
    for exp in experiments:
        MASTER = ExpMaster(exp)
        table = MASTER.load_exps_info(load=old_exp)
        final_table.append(table)

    final_table = pd.concat(final_table)
    save_to_csv(final_table,table_path)

def plot_best_elevation_angles(exps,table_path,show=False):
    best = get_best_exps(table_path,exps)
    for sub,exp in zip(best,exps):
        new_exp = ExpAnalyser(exp,sub,load=True)
        if exp == 7 :
           new_exp.plot_error_less_then_1_image('test',save=True,show=show)
        else:
           uni = new_exp.plot_error_less_then('test',param='delta_elevation',show=show)
    return uni
    
def find_correlation(exps,params,table_path,new_path):
    table = pd.read_csv(table_path)
    cor_table = {'exp': []}
    for param in params:
        cor_table[param] = []
    for exp in exps:
        exp_table = table[table['Experiment']==exp]
        cor_table['exp'].append(exp)
        for param in params:
            x = np.array(exp_table[param])
            if param=='Sub_exp':
                x[np.argwhere(np.isnan(x))] = 0
            y = np.array(exp_table['test_score'])
            R = np.corrcoef(x,y)[1,0]
            cor_table[param].append(R)

    cor_table = pd.DataFrame(cor_table)
    cor_table.columns = cor_table.columns.astype(str)
    save_to_csv(cor_table,new_path)

def advanced_correlation(exps,params,table_path,new_path):
    table = pd.read_csv(table_path)
    tot_combos = []
    cor_table = {'exp': []}
    params_options_array  = {}
    for param in params:
        params_options = np.unique(table[param])
        params_options_array[param] = params_options

    combinations_lengths = np.arange(1,len(params)+1,1)
    for combination_len in combinations_lengths:
        n = len(params)
        k = combination_len
        combos = np.array([[1 if i in comb else 0 for i in range(n)]
        for comb in combinations(np.arange(n), k)])
        for combo in combos:
            tot_combos.append(combo)
            header = str(np.array(params)[np.where(combo)])
            cor_table[header] = []
    
    

    for exp in exps:
        exp_table = table[table['Experiment']==exp]
        indexes = []
        cor_table['exp'].append(exp)
        
        for i in range(len(exp_table)):
            index = []
            for param in params:
                param_val = exp_table.iloc[i][param]
                temp_ind = np.where(params_options_array[param]==param_val)[0][0]
                index.append(temp_ind)
            indexes.append(index)
        exp_table['indexes'] = indexes

        for combo in tot_combos:
            header = str(np.array(params)[np.where(combo)])
            sub_indexes = []
            for i in range (len(exp_table['indexes'])):
                sub_indexes.append(np.array(exp_table.iloc[i]['indexes'])[np.where(combo)])
                unique_groups = np.unique(np.array(sub_indexes),axis=0)
            x = np.zeros(len(exp_table))
            sub_indexes = np.array(sub_indexes)
            
            for i in range (len(unique_groups)):
                match = np.arange(len(exp_table))
                for j in range(len(unique_groups[0])):
                    temp = np.where(sub_indexes[:,j]==unique_groups[i][j])
                    match = np.intersect1d(temp,match)
                x[match] = i  
                    
            y = np.array(exp_table['test_score'])
            R = np.corrcoef(x,y)[1,0]
            cor_table[header].append(R)

    cor_table = pd.DataFrame(cor_table)
    cor_table.columns = cor_table.columns.astype(str)
    save_to_csv(cor_table,new_path)

def get_database_exp(table_path,case_studie,save = True):
    best_case_studie= get_best_exps(table_path,case_studie)
    case_studie_exp = ExpAnalyser(case_studie[0],best_case_studie[0],load=True)
    if save:
        dataframe = case_studie_exp.test_data
        dataframe = pd.DataFrame(dataframe)
        dataframe.columns = dataframe.columns.astype(str)
        dataframe.reset_index(drop=True)
        dataframe.to_feather(case_studie_test)
        dataframe = case_studie_exp.train_data
        dataframe = pd.DataFrame(dataframe)
        dataframe.columns = dataframe.columns.astype(str)
        dataframe.reset_index(drop=True)
        dataframe.to_feather(case_studie_train)
        dataframe = case_studie_exp.valid_data
        dataframe = pd.DataFrame(dataframe)
        dataframe.columns = dataframe.columns.astype(str)
        dataframe.reset_index(drop=True)
        dataframe.to_feather(case_studie_valid)
    return case_studie_exp


    
def images_contribution(case_studie, exps,table_path):
    case_studie_exp = get_database_exp(table_path,case_studie,save = False)
    best = get_best_exps(table_path,exps)

    two_image_model = ExpAnalyser(exps[0],best[0],load=True)
    df = two_image_model.test_data.raw_data_indexes
    two_image_model_raw_data = np.concatenate(df.values).reshape(df.shape[0],-1)


    one_image_model = ExpAnalyser(exps[1],best[1],load=True)
    df = one_image_model.test_data.raw_data_indexes
    one_image_model_raw_data = np.concatenate(df.values).reshape(df.shape[0],-1)

    
    scores_2 = []
    scores_1 = []
    for i in range(len(case_studie_exp.test_data)):
        raw_index = case_studie_exp.test_data.raw_data_indexes[i]
        two_img_eq_ind = np.where((two_image_model_raw_data[:,0]==raw_index[0]) & (two_image_model_raw_data[:,1]==raw_index[1]))[0][0]
        two_img_score = two_image_model.test_data.angles[two_img_eq_ind]
        scores_2.append(two_img_score)
        one_img_eq_ind = np.where(one_image_model_raw_data==raw_index[0])[0][0]
        one_img_score = one_image_model.test_data.angles[one_img_eq_ind]
        scores_1.append(one_img_score)

    dataframe = case_studie_exp.test_data
    dataframe['scores_1'] = scores_1
    dataframe['scores_2'] = scores_2
    dataframe = pd.DataFrame(dataframe)
    dataframe.columns = dataframe.columns.astype(str)
    dataframe.reset_index(drop=True)
    dataframe['cont13'] = (dataframe.scores_1 - dataframe.angles)
    dataframe['cont23'] = (dataframe.scores_2 - dataframe.angles)
    dataframe['cont12'] = (dataframe.scores_1 - dataframe.scores_2)
    dataframe.to_feather(image_cont)
    return dataframe

def analyze_images_contribution(case_studie,exps,table_path,recreate_table = False):
    if recreate_table:
        image_cont_array = images_contribution(case_studie, exps,table_path)
    
    else:
        image_cont_array = feather.read_dataframe(image_cont)

    con_list = ['cont12','cont23','cont13']
    posColor = '#7AC5CD'
    negColor = '#F08080'
    for cont in con_list:
        plt.figure()

        data = image_cont_array[cont]
        N, bins, patches = plt.hist(data,bins=100,color = posColor  ,density=True)
        z_bin = abs(bins).argmin()
        plt.hist([],bins=1,color = posColor, label='Accuracy Improved' )

        plt.hist([],bins=1,color = negColor, label='Accuracy Degraded' )
        # myColor = '#66CDAA'	
        

        for i in range(z_bin):
            patches[i].set_facecolor(negColor)
            
        # plt.hist(data[data<0],bins=100,color='blue',density=True)
        # plt.hist(data[data>0],bins=100,color='green',density=True)
        med  = statistics.median(data)
        above_zero = np.round(len(data[data>0])/len(data)*100,2)
        mu, std = norm.fit(data) 
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [posColor,negColor]]
        labels= ["pos","neg"]
        

        
        plt.plot(x, p, 'k', linewidth=2)
        plt.axvline(x= mu, ymin = 0, linestyle='dotted', color='red',label='Mean')
        plt.axvline(x= mu+std, ymin = 0, linestyle='dotted', color='blue',label='Standard deviation')
        plt.axvline(x= mu-std, ymin = 0, linestyle='dotted', color='blue')
        plt.axvline(x= med, ymin = 0, linestyle='dotted',color='green', label='Median')

        plt.grid(which='both')
        plt.xlabel('Accuracy contribution ['+u'\N{DEGREE SIGN}'+']')
        plt.ylabel('[AU]')
        extraticks=[mu, mu+std, mu-std]
        plt.xticks([-50,50])
        plt.xticks(list(plt.xticks()[0]) + extraticks)
        plt.xlim(-50,80)
        # plt.axvspan(0, max(data), color='blue', alpha=0.1)
        title = cont
        plt.legend()
        # plt.legend(handles, labels)
        plt.savefig('Analayse_data\\cont\\'+title+'_impr_per='+str(above_zero)+'.png',orientation = 'portrait')

def decision_model_collect_data():
    dataset_files = [case_studie_train,case_studie_valid,case_studie_test]
    for dataset_file,desicion_path in zip(dataset_files,desicion_paths):
        database = feather.read_dataframe(dataset_file)
        df = database.raw_data_indexes
        raw_data_indexes = np.concatenate(df.values).reshape(df.shape[0],-1)
        unique_first_inds = np.unique(raw_data_indexes[:,0])
        joints_1 = []
        joints_2 = []
        E0 =[]
        E1 =[]
        for i in range(len(unique_first_inds)):
            unique_ind = unique_first_inds[i]
            database_img_inds = np.where(raw_data_indexes[:,0]==unique_ind)[0]
            database_img_angles = (np.array(database.angles)[database_img_inds])
            best_elev0_ind = database_img_inds[database_img_angles.argmin()]
            best_e0 = database.delta_elevation[best_elev0_ind][0]
            j_1 = database.inputs[best_elev0_ind][0:40]
            unique_second_inds = np.unique(raw_data_indexes[database_img_inds][:,1])

            for j in range(len(unique_second_inds)):
                unique_second_ind = unique_second_inds[j]
                database_imgs_inds = np.where((raw_data_indexes[:,0]==unique_ind)&(raw_data_indexes[:,1]==unique_second_ind))[0]
                database_imgs_angles = np.array(database.angles)[database_imgs_inds]
                best_elev1_ind = database_imgs_inds[database_imgs_angles.argmin()]
                best_e1 = database.delta_elevation[best_elev1_ind][1]
                j_2 = database.inputs[best_elev1_ind][40:80]
                joints_1.append(j_1)
                joints_2.append(j_2)
                E0.append(best_e0)
                E1.append(best_e1)
        
        dataframe = {'joints_1': [], 'joints_2': [], 'best_e0': [], 'best_e1': []}
        dataframe['joints_1'] = joints_1
        dataframe['joints_2'] = joints_2
        dataframe['best_e0'] = E0
        dataframe['best_e1'] = E1
        dataframe = pd.DataFrame(dataframe)
        dataframe.columns = dataframe.columns.astype(str)
        dataframe.reset_index(drop=True)
        dataframe.to_feather(desicion_path)

def get_dec_data(iter=1):
    if iter==1:
        train_database = feather.read_dataframe(decision_train_data)
        df = train_database.joints_1
        train_joints_1 = np.concatenate(df.values).reshape(df.shape[0],-1)
        best_e0_train = np.array(train_database.best_e0)

        test_database = feather.read_dataframe(decision_test_data)
        df = test_database.joints_1
        test_joints_1 = np.concatenate(df.values).reshape(df.shape[0],-1)
        best_e0_test = np.array(test_database.best_e0)

        return train_joints_1,best_e0_train,test_joints_1,best_e0_test

    if iter==2:
        train_database = feather.read_dataframe(decision_train_data)
        df_1 = train_database.joints_1
        df_2 = train_database.joints_2
        train_joints_1 = np.concatenate(df_1.values).reshape(df_1.shape[0],-1)
        train_joints_2 = np.concatenate(df_2.values).reshape(df_2.shape[0],-1)
        train_joints_12 = np.concatenate([train_joints_1,train_joints_2],axis=1)
        best_e1_train = np.array(train_database.best_e1)

        test_database = feather.read_dataframe(decision_test_data)
        df_1 = test_database.joints_1
        df_2 = test_database.joints_2
        test_joints_1 = np.concatenate(df_1.values).reshape(df_1.shape[0],-1)
        test_joints_2 = np.concatenate(df_2.values).reshape(df_2.shape[0],-1)
        test_joints_12 = np.concatenate([test_joints_1,test_joints_2],axis=1)
        best_e1_test = np.array(test_database.best_e1)

        return train_joints_12,best_e1_train,test_joints_12,best_e1_test



def plot_leaves_data(data):
    f = pd.read_csv(data)
    l = f.surface
    l = np.array(l.dropna())
    plt.hist(l*0.01,bins=5)
    plt.xlabel('Leaf Surface [$Cm^2$]')
    plt.ylabel('No. Leaves')
    plt.grid()
    plt.savefig('numleaves.png')

    f=0