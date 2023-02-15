import pandas as pd
import numpy as np
import itertools
from itertools import permutations
import feather
import params_update
from numpy import cos, sin, pi
from scipy.spatial.transform import Rotation as R
import os
import regex as re

all_data = feather.read_dataframe(params_update.Total_filltered_data)

def fix_matrix_array(load):

    # fix_matrixes = []
    fix_matrix = np.identity(3)
    # for i in range (33):
        # fix_matrixes.append(unit_matrix)

    for i in range (len(load)):
        # print(load['distance'][i],load['elevation'][i],load['rotation'][i],load['orientation'][i])
        if np.round(load['distance'][i],2) ==  0.3:
            if load['elevation'][i] == 0.0:
                if load['rotation'][i] == 0.0:
                    if load['orientation'][i] == 0.0:

                        b = np.stack(load.iloc[i]['joints']).squeeze()
                        # print(b)
                        x = b[0::2]
                        y = b[1::2]
                        # print('x',x)
                        # print('y',y)
                        # x= b[:,0]
                        # y= b[:,1]
                        m, b = np.polyfit(x, y, 1)
                        fix_angle= np.pi/2+ np.arctan(m)
                        fix_matrix = np.array([[1, 0, 0],
                        [0, cos(-fix_angle), -sin(-fix_angle)],
                        [0, sin(-fix_angle), cos(-fix_angle)]])   
                        # print(load['leaf_number'][i])
                        # fix_matrixes[load['leaf_number'][i]] = fix_matrix
                        print('leaf_done:' ,load['leaf_number'][i])
                        # print('fixed',fix_matrix)
    return fix_matrix

def get_parameters(load,index):
    leaf_number = load['leaf_number'][index]
    distance = load['distance'][index]
    elevation = load['elevation'][index]
    rotation = load['rotation'][index]
    orientation = load['orientation'][index]
    return leaf_number,distance,elevation,rotation,orientation

def calc_rotation_mat(elevation,rotation,orientation):
    elevation = elevation*pi/180
    rotation = (2*pi/12) * rotation
    orientation = orientation * pi/4
    T01 = (R.from_euler('x', -elevation)).as_matrix()
    # T12 = UNIT
    T23 = (R.from_euler('z', -rotation)).as_matrix()
    T34 = (R.from_euler('z', pi/2)).as_matrix()
    T4leaf = (R.from_euler('y', orientation)).as_matrix()
    T0leaf = np.dot(T01,T23)
    T0leaf = np.dot(T0leaf,T34)
    T0leaf = np.dot(T0leaf,T4leaf)
    return T0leaf

def get_fix_matrix(fix_matrixes,leaf_number):
    return fix_matrixes[leaf_number]
    
def get_perfect_rot_matrix(T0leaf, fix_matrix):
    new_mat = np.dot(T0leaf,fix_matrix)
    normal = np.dot(new_mat,[0,0,1])
    return new_mat, normal

def main_find_normals(feathername, target_directory, new_name = 'Total_data'):

    load_all = feather.read_dataframe(feathername)
    load_all['leaf_new_orientation_matrix']= ""
    load_all['leaf_normal'] = ""
    fix_array = fix_matrix_array(load_all)
    for i in range (len(load_all)):
        leaf_number,distance,elevation,rotation,orientation = get_parameters(load_all,i)
        T0leaf = calc_rotation_mat(elevation,rotation,orientation)
        # fix_matrix = get_fix_matrix(fix_array,leaf_number)
        rotation_matrix ,normal = get_perfect_rot_matrix(T0leaf, fix_array)
        # print('rotation_matrix',rotation_matrix)
        load_all['leaf_new_orientation_matrix'][i] = rotation_matrix.flatten()
        load_all['leaf_normal'][i] = normal

    # name = str(new_name)
    name = target_directory + new_name
    print('name',name)
    load_all.to_feather(name)


def make_reg_data(all_data):
    data_frame = {'leaf_normal': [], 'Input': [], 'leaf_number': [],'index':[]}
    
    for i in range (len(all_data)):
        data_frame['leaf_normal'].append(all_data['leaf_normal'][i])
        data_frame['leaf_number'].append(all_data['leaf_number'][i])
        data_frame['index'].append(i)
        data_frame['Input'].append(all_data['joints'][i])
    
    data_frame = pd.DataFrame(data_frame)
    data_frame.columns = data_frame.columns.astype(str)
    data_frame.reset_index(drop=True)
    data_frame.to_feather('combos_1.feather')
    print('saving combos_1.feather')


def prepare_by_joints_num_reg(n,load):

    print('creating file: ' + str(n) + ' joints for reg images')
    data_frame = {'leaf_number': [],'leaf_normal': [], 'input': [], 'index': [], 'old_input': []}
    for index in range(len(load)):
        img_joints = load['Input'][index]
        _,_,joints = get_eq_spaced_pixels(img_joints,n)
        data_frame['input'].append(np.array(joints).flatten())
        data_frame['leaf_normal'].append(load['leaf_normal'][index])
        data_frame['leaf_number'].append(load['leaf_number'][index])
        data_frame['index'].append(load['index'][index])
    
    data_frame['old_input'] = load['Input']
    name = 'combos_1_' + str(n) + '_joints.feather'
    data_frame = pd.DataFrame(data_frame)
    data_frame.columns = data_frame.columns.astype(str)
    data_frame.reset_index(drop=True)
    data_frame.to_feather( name)
    print('file ',name,' is saved')

def get_eq_spaced_pixels(img_joints,n,min_x=0,min_y=0,first_im=True):
    x_pixels = img_joints[0::2]
    y_pixels = img_joints[1::2]
    idx = np.round(np.linspace(0, len(x_pixels) - 1, n)).astype(int)
    x_pixels = x_pixels[idx]
    y_pixels = y_pixels[idx]
    if first_im:
        min_x = np.min(x_pixels)
        min_y = np.min(y_pixels)

    x_pixels = x_pixels-min_x
    y_pixels = y_pixels-min_y

    new_joints = [item for pair in zip(x_pixels, y_pixels) for item in pair]
    if first_im:
        return min_x,min_y,np.array(new_joints).flatten() 

    return np.array(new_joints).flatten()  


def check_conditions(load,i,j):
    check = False
    if load['leaf_number'][i]==load['leaf_number'][j]:
        if load['rotation'][i]==load['rotation'][j]:
            if load['orientation'][i]==load['orientation'][j]:
                check = True
    return check


def make_total_filltered_pairs(load):
    data_frame = {'leaf_number': [],'leaf_normal': [], 'index_1': [], 'index_2': []}
    for i in range (len(load)):
    # for i in range (10):
        # print(i)
        
        for j in range (len(load)):
            if (i!=j):
                check = check_conditions(load,i,j)
                
                if check == True:
                    # print(load.iloc[i],load.iloc[j])
                    data_frame['index_1'].append(i)
                    data_frame['index_2'].append(j)
                    data_frame['leaf_normal'].append(load['leaf_normal'][j])
                    data_frame['leaf_number'].append(load['leaf_number'][j])
                    
    data_frame = pd.DataFrame(data_frame)
    data_frame.columns = data_frame.columns.astype(str)
    data_frame.reset_index(drop=True)
    data_frame.to_feather(params_update.Total_filltered_pairs)


def combine_n_images(pairs_indexes_file,n,all_data):
    unique_start_indexes = pairs_indexes_file['index_1'].unique()
    
    print('combining images...')
    all_combos = []
    count = 0
    for start_index in unique_start_indexes:
        count+=1
        if ((count%100)==0):
            print('start index # ',count,'/',len(unique_start_indexes))
        pos_pairs = pairs_indexes_file[pairs_indexes_file['index_1']==start_index]['index_2']
        unique_combos=itertools.combinations(pos_pairs, n-1)
        g=np.array(list(unique_combos))

        for index in range (len(g)):
            combos_for_row=np.array(list(permutations(g[index])))
            new_arr=np.zeros((np.shape(combos_for_row)[0],np.shape(combos_for_row)[1]+1))
            new_arr[0:combos_for_row.shape[0], 1:1+combos_for_row.shape[1]] = combos_for_row
            new_arr[:,0] = start_index
            all_combos=np.append(all_combos,new_arr.astype(int))

    print('done calculating images combinations...')

    all_combos = all_combos.reshape(-1,n).astype(int)
    data_frame = {'leaf_normal': [], 'Input': [], 'leaf_number': [],'index':[]}
    for i in range (n):
        data_frame[str('index_'+str(i+1))]= []

    count = 0
    
    num_files=10
    saving_points = np.linspace(len(all_combos)/num_files,len(all_combos),num_files).astype(int)
    file_num = 0
    for combo in all_combos:
        count+=1
        if ((count%100)==0):
            print('progress ', count, '/',len(all_combos))
        for i in range (n):
            data_frame[str('index_'+str(i+1))].append(combo[i])
            
        data_frame['leaf_normal'].append(all_data['leaf_normal'][combo[-1]])
        data_frame['leaf_number'].append(all_data['leaf_number'][combo[-1]])
        data_frame['index'].append(count-1)
        data_frame['Input'].append(get_input(combo,all_data))

        if n==3:
            if count == saving_points[file_num]:
                file_num+=1
                print('saving file num: ',file_num, '/', len(saving_points))
                data_frame = pd.DataFrame(data_frame)
                data_frame.columns = data_frame.columns.astype(str)
                data_frame.reset_index(drop=True)
                data_frame.to_feather('combos_' +str(n)+'_'+str(file_num)+'.feather')
                print('combos_' +str(n)+'_'+str(file_num)+'.feather')
                data_frame = {'leaf_normal': [], 'Input': [], 'leaf_number': [],'index':[]}
                for i in range (n):
                    data_frame[str('index_'+str(i+1))]= []
    
    if n!=3:
        data_frame = pd.DataFrame(data_frame)
        data_frame.columns = data_frame.columns.astype(str)
        data_frame.reset_index(drop=True)
        data_frame.to_feather('combos_' +str(n)+'.feather')
        print('combos_' +str(n)+'.feather')


def get_input(indexes,all_data):
    joints = all_data['joints'][indexes[0]]
    delta = all_data['end_effector_pose'][indexes[1]]-all_data['end_effector_pose'][indexes[0]]
    delta = delta[0]
    for i in range (len(indexes)-1):
        new_joints = all_data['joints'][indexes[i+1]]
        joints= np.append(joints,new_joints)
        try:
            new_delta = all_data['end_effector_pose'][indexes[i+2]]-all_data['end_effector_pose'][indexes[i+1]]
            delta = np.append(delta,new_delta[0])
        except:
            input = np.append(joints,delta)
            return input

def prepare_by_joints_num(n,load,img_num,sub_num = False):
    
    start_index = [0]
    
    total_joints = (len(load['Input'][0])-6*(img_num-1))/(2*img_num)

    img_joints = [] 
    for i in range(img_num-1):
        start_index.append((i+1)*total_joints*2)
    print('creating file: ' + str(n) + ' joints')
    data_frame = {'leaf_number': [],'leaf_normal': [], 'input': [], 'index': [], 'old_input': []}

    for index in range(len(load)):
        joints = []
        for start in start_index:
            img_joints = load['Input'][index][int(start):int(start+total_joints*2)]
            if start==0:
                min_x,min_y,new_joints = get_eq_spaced_pixels(img_joints,n)
            else:
                new_joints = get_eq_spaced_pixels(img_joints,n,min_x,min_y,first_im=False)
            joints=np.append(joints,new_joints)
        delta = load['Input'][index][-6*(img_num-1):]

        new_input =  np.append(np.array(joints).flatten(),delta)
        data_frame['input'].append(new_input)
        data_frame['leaf_normal'].append(load['leaf_normal'][index])
        data_frame['leaf_number'].append(load['leaf_number'][index])
        data_frame['index'].append(load['index'][index])

    data_frame['old_input'] = load['Input']
    name = 'combos_'+ str(img_num)+'_' + str(n) + '_joints.feather'
    if sub_num:
        name = 'combos_'+ str(img_num)+'_' + str(n) + '_joints'+'_'+str(sub_num)+'.feather'

    data_frame = pd.DataFrame(data_frame)
    data_frame.columns = data_frame.columns.astype(str)
    data_frame.reset_index(drop=True)
    data_frame.to_feather( name)
    print('file ',name,' is saved')

def filtering(load,new_name):
    trash_list = get_trash_files()
    unwanted = get_trash_list(trash_list)
    delete_unwanted(unwanted,load, new_name)

def get_trash_files():
    # num_of_folders = 32
    Trash_directory = params_update.trash_directory
    trash_list = []
    for filename in os.listdir(Trash_directory):
        if filename.endswith('.png'):
            # trash_list.append(filename[:-16])
            trash_list.append(filename)
    return trash_list

def parameters_from_filename(filename):
    pattern = "N(.*?)_d"
    leaf_number = re.search(pattern, filename).group(1)
    pattern = "d(.*?)_p"
    distance = re.search(pattern, filename).group(1)
    pattern = "p(.*?)_r"
    elevation = re.search(pattern, filename).group(1)
    pattern = "r(.*?)_O"
    rotation = re.search(pattern, filename).group(1)
    pattern = "O(.*?).png.feather"
    orientation = re.search(pattern, filename).group(1)
    parameters = np.float16([leaf_number,distance,elevation,rotation,orientation])
    return parameters

def get_trash_list(trash_list):
    unwanted = []
    for i in range (len(trash_list)):
        unwanted.append(parameters_from_filename(trash_list[i]))
    return unwanted

def delete_unwanted(unwanted_list,load, new_name):
    how_many_found = 0
    for i in range(len(unwanted_list)):
        # print('num unw',i)
        leaf_number,distance,elevation,rotation,orientation = unwanted_list[i]
        # print(unwanted_list[i])

        load,count = delete_by_parameters(load,leaf_number,distance,elevation,rotation,orientation)
        how_many_found=how_many_found+count
    load.to_feather(new_name)
    print('Number of files filltered: ', how_many_found)   

def delete_by_parameters(load,leaf_number,distance,elevation,rotation,orientation):
    count=0
    for i in range (len(load)):

        if load['leaf_number'][i] ==  leaf_number:
            # print(load['distance'][i],distance)
            if abs(load['distance'][i]-distance)<0.002:
                if load['elevation'][i] == elevation:
                    if load['rotation'][i] == rotation:
                        if load['orientation'][i] == orientation:
                            load = load.drop([i])
                            load = load.reset_index(drop=True)
                            # print('yes')
                            count = 1
                            return load,count
    return load, count                      

def combine_feathers():
    directory = params_update.norms_and_joints_folder
    new_file = []
    indexes = []
    for file in os.listdir(directory):
        if file.endswith('.feather'):
            filename = os.path.join(directory, file)
            load = feather.read_dataframe(filename)
            indexes= np.append(indexes, np.array(load.index).flatten())
            # indexes = np.array(indexes).flatten()
            # print(indexes)
            pandas = pd.DataFrame(load)
            # print(load)
            new_file.append(pandas)
    # new_file = pd.DataFrame(new_file)
    # print(new_file)
    new_file = pd.concat(new_file)
    # print(np.array(indexes).flatten())
    new_file['index_raw_data'] = np.array(indexes).flatten()
    new_file.columns = new_file.columns.astype(str)
    
    new_file.reset_index(drop=True).to_feather(params_update.Total_data)

def find_normals():

    for file in os.listdir(params_update.joints_folder) :
        if file.endswith('.feather'):
            feathername = os.path.join(params_update.joints_folder, file)
            new_name = 'normals_and_'+ file
            main_find_normals(feathername, params_update.norms_and_joints_folder, new_name)


def creat_complete_file(folders):
    # directory = 'C:\data_exp\data\\'
    directory = params_update.exp_data
    # for folder in os.listdir(directory):
    for folder in folders:
        new_file = []
        parameters = {'leaf_number': [] ,'distance': [] ,'elevation': [] ,'rotation': [] ,'orientation': [] }
        print('folder number =', folder)
        folder_path = os.path.join(directory, str(folder))
        i=-1
        for file in os.listdir(folder_path):
            if file.endswith(".feather"):
                i = i+1
                file_path = os.path.join(folder_path, file)
                leaf_number,distance,elevation,rotation,orientation = parameters_from_filename(file)
                # print(leaf_number,distance,elevation,rotation,orientation)
                load = feather.read_dataframe(file_path).iloc[0]
                new_file.append(load)
                parameters['leaf_number'].append(leaf_number)
                parameters['distance'].append(distance)
                parameters['elevation'].append(elevation)
                parameters['rotation'].append(rotation)
                parameters['orientation'].append(orientation)
        
        new_file = pd.DataFrame(new_file)
        new_file.columns = new_file.columns.astype(str)
        new_file['index'] = range(0, len(new_file))
        new_file = new_file.set_index('index')
        parameters = pd.DataFrame(parameters)
        parameters.columns = parameters.columns.astype(str)
        new_file = pd.concat([new_file, parameters.reindex(new_file.index)], axis=1)
        # new_file = new_file.append(parameters)
        data_frame = pd.DataFrame(new_file)
        data_frame.columns = data_frame.columns.astype(str)
        name = params_update.raw_data_directory+'//rawdata_'+ str(folder)+'.feather'
        if data_frame.empty:
            print('folder', folder, 'is empty')
        else:
            data_frame.reset_index(drop=True).to_feather(name)
            print('saved:', name)
