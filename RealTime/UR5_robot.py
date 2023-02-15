from regex import E
from camera import Camera
from NNmodel import RealTimeModel
import socket
import numpy as np
from ur5_kin_tester import UR5_KIN
from ur5_tester import UR5_COM
import time
from utils import draw_circles, get_green_img, get_center,get_leaf_surface,get_rotational_path,conv_vec,convert_list,find_vec_in_ef_system,error_angle,find_normals
from utils import rot_around_z,rot_around_x,find_angle,rot_around_y, norm_pixels,draw_normal_on_image,reject_outliers,plot_3d_vectors,fix_vecs,get_params,get_fix_matrix
from utils import change_direction_joints
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import math
import feather
import os
import itertools
import cv2 as cv
import pandas as pd
import regex as re
from itertools import permutations

zeroes = True

class Robot:
    def __init__(self,
                steps = 3,
                realtime = True,
                img_folder = 'images//',
                test_folder = 'images//test//',
                new_test_folder = 'data//',
                r_plant = 0.3,
                fake_robot = False,
                fake_camera = False,
                current_measure = True,
                num_exp = 5,
                sub_exp = 5
                ):
        self.img_folder = img_folder
        self.test_folder = test_folder
        self.new_test_folder = new_test_folder
        self.combos_calculated = []
        self.norm_list = []
        self.ef_list = []
        self.r_plant = r_plant
        self.current_measure = current_measure
        self.fake_robot = fake_robot
        self.fake_camera = fake_camera
        self.ef_pose_1 = []
        self.ef_pose_2 = []
        self.ef_pose_3 = []
        self.num_exp =num_exp
        self.sub_exp =sub_exp
        self.no_ef = False
        if self.num_exp>7:
            self.no_ef = True

        if not self.fake_robot:
            self.connect_ur5()
        try:
            print('Connecting Camera...')
            self.camera = Camera(self.fake_camera)
            print('Camera Connection Accomplished')
        except:
            print('Camera Connection Failed')
        self.n_steps = steps
        self.ur5_kin = UR5_KIN() # for calculating kinetics
        self.ur5_kin.d_base = [1,-1]
        self.ur5_kin.r_plant = -0.5
        self.start_rot = -np.pi*0.25
        self.start_tcp = 0
        self.th_tcp_steps = 5
        self.th_tcp_range = [-30,30]

        
        self.load_model()
    
    def load_model(self):

            
        self.model = RealTimeModel(INPUT_SIZE = (132-12*self.no_ef),
                num_exp = self.num_exp,
                sub_exp = self.sub_exp)
        
    
    def connect_ur5(self,isprint=True):
        if not self.fake_robot:
            if isprint:
                print('Connecting UR5...')
            HOST = "192.168.1.113"
            PORT_30003 = 30003
            try:
                self.ur5 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.ur5.settimeout(5)
                self.ur5.connect((HOST, PORT_30003))
                self.ur5_com = UR5_COM(self.ur5)
                if isprint:
                    print('UR5 Connection Accomplished')
                self.joints_pose = self.ur5_com.get_joints_pose()
                self.tcp_pose = self.ur5_com.get_tcp_position()
                return self.tcp_pose, self.joints_pose
            except:
                if isprint:
                    print('UR5 Connection Failed')

    

    def find_leaf_center(self,index=0,realtime=False,fake_camera=False,exmp = 0):
        temp_name = 'temp.png'
        if not fake_camera:
            self.camera.play_camera()
        else:
            img = feather.read_dataframe(exmp)
            self.camera.color_image = np.array(img['img_color'])[0].reshape(np.array(img['shape_color'])[0])
 
        plt.imsave(temp_name,self.camera.color_image)
        temp_img = cv.imread(temp_name)
        time.sleep(3)
        temp_img[0:250,0:680] = [0,0,0]
        temp_img[0:480,600:680] = [0,0,0]
        _,self.green = get_green_img(temp_img)
        self.start_joints, self.center = get_center(self.green)
        self.found_joints = min(max(self.start_joints[:,1])-min(self.start_joints[:,1]),max(self.start_joints[:,0])-min(self.start_joints[:,0]))>20


    
    def move_x_y(self,norm_vec,step_size):
        self.connect_ur5(isprint=False)
        temp_move = self.tcp_pose
        new_vec = np.linalg.norm(np.dot(self.Ef_rot_mat,[norm_vec[0],norm_vec[1],0,0])[0:3])
        temp_move[0:3] = self.tcp_pose[0:3]+new_vec*step_size
        try:
            self.move(temp_move,type = 'pose')
            self.connect_ur5(isprint=False)
        except:
            print('step_not_done')

    def move_axis(self,axis,step_size):
        self.connect_ur5(isprint=False)
        if axis == 'x':
            vec = self.x_vec
        if axis == 'y':
            vec = self.y_vec
        if axis == 'z':
            vec = self.z_vec
        temp_move = self.tcp_pose
        temp_move[0:3] = self.tcp_pose[0:3]+vec*step_size
        try:
            print('Moving ',step_size,' ',axis,' axis' )
            self.move(temp_move,type = 'pose')
            time.sleep(3)
        except:
            print('step_not_done')

    def find_cur_dis(self):
        cur_dis_y = self.center[0] - 320
        cur_dis_x = 240 - self.center[1]
        vec = [cur_dis_x,cur_dis_y]
        cur_dis = np.linalg.norm(vec)
        norm_vec = vec/cur_dis
        return cur_dis, norm_vec

    def move_to_center(self):
        print('Moving Camera to Leaf Center')
        

        self.connect_ur5(isprint=False)
        step_size_pos = -0.05
        step_size = step_size_pos
        cur_dis, norm_vec = self.find_cur_dis()
        if cur_dis<80:
            step_size = 0.7*step_size
            
        ind = 0
        while (cur_dis>50) :
            ind+=1
            print('moving ', np.round(norm_vec*step_size,3),'[m] X,Y axis')
            self.move_x_y(norm_vec,step_size)
            prev_center = self.center
            # print('prev_center',np.round(prev_center,2))
            temp_old_dis = cur_dis
            found = False
            frac = 0.5
            while not found:
                try:
                    self.find_leaf_center(index=ind)
                    new_center = self.center
                    if (abs(new_center[0]-prev_center[0])<30):
                        time.sleep(5)
                        self.find_leaf_center(index=ind) #Try again
                    found=True
                    if not self.found_joints:
                        print('Joints not found')
                        self.move_x_y(norm_vec,-step_size) # return and try again
                except:
                    self.move_x_y(norm_vec,-step_size) # move backwards
                    step_size = 0.95*step_size
                    print('Moving Backwards')
                    frac = frac*1.5
                    
            # print('new_center',self.center)
            old_norm_vec = norm_vec
            cur_dis, norm_vec = self.find_cur_dis() 
            print('Prev Distance:', np.round(temp_old_dis,2))
            print('Current Distance:', np.round(cur_dis,2))
            time.sleep(3)
            if abs(cur_dis-temp_old_dis)<10: #you didnt do anything
                cur_dis, norm_vec = self.find_cur_dis() 

            elif cur_dis>temp_old_dis: #go back
                print('Changing direction')
                if abs(cur_dis-temp_old_dis)>100: #movement was too big
                    step_size = -0.95*step_size
                if abs(cur_dis-temp_old_dis)<50: #movement was too small
                    step_size = -step_size
                elif cur_dis>200: #you are very far
                    if temp_old_dis<80: #but you were very close
                        step_size = -step_size #return and try again
                elif cur_dis<100: #you are very close
                    step_size = -0.8*step_size_pos
                
                else:
                    step_size = -0.75*step_size
                    norm_vec = old_norm_vec
            else: #you got closer
                # if cur_dis>150: #you are still far away
                #     if abs(cur_dis-temp_old_dis)<50: #you progress slowly:
                #         step_size = 3*step_size #move faster
                if cur_dis<80: #you are very close
                    step_size = 0.8*step_size_pos
                
        self.find_leaf_center(index = 'finish',save=True)
        
        print('Finding leaf x,y center complete')


    def change_dis(self):
        cur_sur = get_leaf_surface(self.start_joints)
        found_dis = False
        sign = 1
        prev_sign = 1
        step_size = 0.2
        # dis = -48300*self.r_plant+27400
        dis = 9000
        buffer = 1000
        min_sur = dis - buffer
        max_sur = dis + buffer
        while not found_dis:
            go_forward = cur_sur<min_sur
            go_back =  cur_sur>max_sur
            sign = go_forward - go_back
            if sign!=prev_sign:
                step_size = 0.5*step_size
            
            self.move_axis('z',sign*step_size)
            
            self.find_leaf_center()
            time.sleep(5)
            cur_sur = get_leaf_surface(self.start_joints)
            print('surface:',np.round(cur_sur,2))
            prev_sign = sign

            if cur_sur>min_sur:
                if cur_sur<max_sur:
                    found_dis = True
        

    def fix_camera_z_axis(self):
        
        mat = self.get_fix_mat()
        fixed_pose = np.zeros(6)
        quat = self.tcp_pose[0:3] # loc stays the same
        fixed_rotation = Rotation.from_matrix(mat[0:3,0:3]).as_rotvec()
        fixed_pose[0:3] = quat
        fixed_pose[3:6] = fixed_rotation
        self.move(fixed_pose,'pose')

        # recalculate all axis
        self.connect_ur5(isprint=False)
        self.start_point = self.tcp_pose
        self.get_fixed_axis() 

    def get_cur_tcp_rot_mat(self,realtime):
        if not self.fake_robot:
            if realtime:
                self.connect_ur5(isprint=False)
        rotvec = self.tcp_pose[3:]
        quat = self.tcp_pose[0:3]
        Ef_rot_mat = np.zeros([4,4])
        Ef_rot_mat[0:3,0:3]= Rotation.from_rotvec(rotvec).as_matrix()
        Ef_rot_mat[0:3,3] = quat
        Ef_rot_mat[3,3] = 1
        return Ef_rot_mat

    def get_th_rot(self,st_point ,realtime=True):
        if realtime:
            self.connect_ur5(isprint=False)
        if self.fake_robot:
            self.tcp_pose = [0,0,0,0,0,0]
        else:
            self.tcp_pose = st_point
        self.get_fixed_axis(realtime=realtime)
        self.th_rot = find_angle(self.y_vec,[0,1,0])
        self.rotation_matrix = rot_around_z(self.th_rot+np.pi*0.5)

    def get_fix_mat(self,realtime=True):
        self.connect_ur5(isprint=False)
        T_0c = self.get_cur_tcp_rot_mat(realtime=realtime)
        x_d = ([0,0,1])
        x_c = np.dot(T_0c[0:3,0:3],[1,0,0])
        y_rot = find_angle(x_d,x_c)
        T_cd = rot_around_y(y_rot)
        T_0d = np.dot(T_0c,T_cd)
        self.camera_pose_mat = T_0d
        return self.camera_pose_mat
 

    def adjust_dis(self,move_to_center=True,change_z = True):

        if move_to_center:
            self.move_to_center()
        if change_z:
            self.change_dis()
            
    def plan_route(self):
        print('planning route for\n Distance:',self.r_plant,'[m]')
        print('Elavation angles:',self.best_route,'[Degrees]')


        if self.r_plant==0.1:
            self.th_tcp_steps = 3
            self.th_tcp_range = [-10,10]
        if self.r_plant==0.2:
            self.th_tcp_steps = 3
            self.th_tcp_range = [-20,20]
        if self.r_plant==0.3:
            self.th_tcp_steps = 5
            self.th_tcp_range = [-30,30]
        self.positions, self.angles = get_rotational_path(self.r_plant,self.d_base,
                                                            self.rotation_matrix, self.th_rot,
                                                            move_type='pose',
                                                            angles = self.best_route)

    def get_fixed_axis(self,realtime=True):
        
        self.Ef_rot_mat = self.get_cur_tcp_rot_mat(realtime=realtime)
        self.x_vec = np.dot(self.Ef_rot_mat,[1,0,0,0])[0:3]
        self.y_vec = np.dot(self.Ef_rot_mat,[0,1,0,0])[0:3]
        self.z_vec = np.dot(self.Ef_rot_mat,[0,0,1,0])[0:3]

    def move_start_point(self,pose):
        print('Moving to Start Pose')
        self.start_point_joints  = pose
        self.move(self.start_point_joints, 'pose') #EndEffector aprox directed in front of the leaf
        time.sleep(10)
        self.connect_ur5(isprint=False)
        time.sleep(5)
        self.start_point = self.ur5_com.get_tcp_position()
        self.get_fixed_axis()
        
    def detect_target_leaf(self):
        axiss = ['z','x','y']
        sign = 1
        index = 0 
        counter = 0
        step_size = 0.05
        steps = [-step_size,2*step_size,-step_size]
        print('Detecting Target Leaf...')
        found = False
        while not found:
            if counter==3:
                counter=0
                index+=1
            if counter==0:
                axis = axiss[index]
            try:
                if counter==2:
                    step = steps[counter]
                    self.move_axis(axis,step) #return
                else:
                    self.find_leaf_center()
                    if self.found_joints:
                        found = True
            except:
                step = steps[counter]
                found = False
                self.move_axis(axis,step)

            counter+=1

    def locate_leaf(self,st_point= 0 ,realtime=True):
        self.get_th_rot(st_point=st_point,realtime=realtime)
        T_01 = self.rotation_matrix
        T_10 = np.linalg.inv(T_01[0:3,0:3])
        self.leaf_location_1 = np.dot(T_10,self.tcp_pose[0:3]) #Leaf location in 1 coordinate system

        # For img taking route planning
        L1 = self.leaf_location_1[0]-0.07
        L2 = self.leaf_location_1[1]+0.3
        # self.leaf_location_1[1] = L1
        L3 = self.leaf_location_1[2] - 0.03
        self.d_base = [L1,L2,L3]

        
        #adjusments for grasping

        change_x = -0.07
        change_z = 0.1
        change_y = 0.25
    

        leaf_loc_1 = self.leaf_location_1
        #change x


        leaf_loc_1[0] = leaf_loc_1[0]+change_x
        #change y
        leaf_loc_1[1] = leaf_loc_1[1]+change_y
        #change z
        leaf_loc_1[2] = leaf_loc_1[2]+change_z
        self.leaf_gripper = leaf_loc_1
        #conv to base
        self.leaf_location = np.dot(T_01[0:3,0:3],leaf_loc_1) 
        l = np.zeros(6)
        l[0:3] = self.leaf_location 
        l[3:6] = self.tcp_pose[3:6]
        print('leaf location:',self.leaf_location )

        
    def start(self,pose,move_to_center,change_z,detect=True): # find leaf, find leaf center, adjust distance, plan route
        if not self.fake_robot:
            self.move_start_point(pose)
            if detect:
                self.detect_target_leaf() 
            self.adjust_dis(move_to_center=move_to_center,change_z=change_z)
        self.locate_leaf()
        

    def take_img(self,index,fake_camera,exmple_file = 0): # take first img and save end effector location
        if not self.fake_robot:
            self.connect_ur5(isprint=False)
        self.found_joints =False
        self.good_img = False
            
        while not (self.found_joints*self.good_img):
            self.find_leaf_center(index=index,fake_camera=fake_camera,exmp=exmple_file)
            time.sleep(5)
            
            if index==0:
                self.prev_img = list(self.camera.color_image.flatten())
                self.cur_img = self.prev_img
                self.good_img = True
                
            else:
                self.cur_img = list(self.camera.color_image.flatten())
                # if len(np.where((self.cur_img['img_color'].all()==self.prev_img['img_color'].all())==False)[0])!=0:
                if len(np.where((self.cur_img==self.prev_img)==False)[0])!=0:
                    self.good_img = True
                    self.found_joints = True
        
        self.prev_img = self.cur_img

    def save_data(self,folder,name):
        self.data_frame = {'img_color': [], 'shape_color': [],'img_green': [],'end_effector_pose': [], 'joints': [] }
        self.data_frame['shape_color'].append(list(self.camera.color_image.shape))
        self.data_frame['img_color'].append(list(self.camera.color_image.flatten()))
        self.data_frame['img_green'].append(list(self.green.flatten()))
        # self.data_frame['shape_depth'].append(list(self.camera.depth_image.shape))
        # self.data_frame['img_depth'].append(list(self.camera.depth_image.flatten()))
        self.data_frame['end_effector_pose'].append(self.tcp_pose)
        self.data_frame['joints'].append(list(self.start_joints.flatten()))
        self.data_frame = pd.DataFrame(self.data_frame)
        self.data_frame.columns = self.data_frame.columns.astype(str)
        new_name = folder+name
        self.data_frame.reset_index(drop=True).to_feather(new_name+'.feather')
        cv.imwrite(new_name+'.png',self.camera.color_image)
        cv.imwrite(new_name+'_green.png',self.green)

        
    def move(self,where,type='joints'):
        if not self.fake_robot:
            xR, yR, zR, rxR, ryR, rzR = where
            attampts = 0
            dis = 80
            while dis> 0.05:
                if attampts<5:
                    attampts+=1
                    if type=='joints':
                        move = ("movej(["+("%f,%f,%f,%f,%f,%f"%(xR, yR, zR, rxR, ryR, rzR)) +"], a=0.15, v=0.2, r=0)" +"\n").encode("utf8")
                        self.ur5.send(move)
                        time.sleep(3)
                        self.connect_ur5(isprint=False)
                        real = self.joints_pose
                    if type=='pose':
                        move = ("movej(p["+("%f,%f,%f,%f,%f,%f"%(xR, yR, zR, rxR, ryR, rzR)) +"], a=0.2, v=0.4, r=0)" +"\n").encode("utf8")
                        self.ur5.send(move)
                        time.sleep(3)
                        self.connect_ur5(isprint=False)
                        real = self.tcp_pose

                    dis = math.dist(real,where)

    def get_done_poses(self):
        poses = []
        rotations = []
        for filename in os.listdir(self.new_test_folder):
            if filename.endswith('.feather'):
                poses.append(filename[:-7])
        self.poses = poses
        self.rotation_of_pose = rotations
       

    def get_all_combos(self):
        combos = []
        for rot in np.unique(self.rotation_of_pose):
            rot_inds = np.where(self.rotation_of_pose==rot)[0]
            for subset_list in itertools.combinations(rot_inds, 3): 
                for subset in list(permutations(subset_list)): 
                    combos.append(subset)
        
        return combos

    def fix_delta(self,delta):
        fixed_delta = np.zeros_like(delta)
        fixed_delta[0:3] = delta[0:3]
        fixed_delta[3] = delta[4]
        fixed_delta[4] = delta[3]
        fixed_delta[5] = -delta[5]
        return fixed_delta



    def flip_joints(self,new_joints):
        k = new_joints.reshape(-1,2)
        t = np.zeros_like(k)
        t[:,0] = k[:,1] 
        t[:,1] = k[:,0]
        # t = new_joints
        return t.flatten()




    def get_input(self,combo):
        joints_all = []
        ef_poses = []
        deltas = []
        count = -1
        first_img=True
        for ind in combo:
            count+=1
            name = self.poses[ind]
            filename = self.new_test_folder+name+'.feather'
            load = feather.read_dataframe(filename)
            img =  np.array(load['img_color'])[0].reshape(load['shape_color'][0])
            plt.imsave('temp_temp.png',img)
            img = cv.imread('temp_temp.png')
            try:
                lower_hsv=np.array([36, 40, 40])
                upper_hsv= np.array([90, 255, 255])
                _,green = get_green_img(img,lower_hsv = lower_hsv, upper_hsv = upper_hsv)
                p = green
                p[0:250,0:680] = [0,0,0]
                p[0:480,600:680] = [0,0,0]
                joints, self.center = get_center(p)
                joints = self.flip_joints(joints)
                

                if first_img:
                    min_x,min_y,new_joints   = norm_pixels(joints,first_im=first_img)
                else:
                    new_joints = norm_pixels(joints,min_x=min_x,min_y=min_y,first_im=first_img)
                
                new_joints = change_direction_joints(new_joints)
                joints_all.append(new_joints)

                ef_poses.append(load['end_effector_pose'][0])
                first_img=False
                if count>0:
                    temp_delta = ef_poses[count] - ef_poses[count-1]
                    temp_delta = self.fix_delta(temp_delta)
                    deltas.append(temp_delta)
                    
            except:
                joints_all.append(0)
                deltas.append(0)
                ef_poses.append(0)

        input = np.array(joints_all).flatten()
        if not self.no_ef:
            input = np.append(input,deltas)
        # if zeroes:
        #     input[120:] = 0
        return input,img,name, ef_poses[-1]

        
    def estimate(self,input):
        output = self.model.estimate(input)
        return output


    def get_vector(self,combo,save = True):
        input,dest_img,name,ef_pose = self.get_input(combo)
        try:
            output = self.estimate(input)
        except:
            output = [[0,0,0]]
        if save:
            new_img = draw_normal_on_image(output[0],dest_img,self.center,vec_len=100)
            plt.imsave(self.new_test_folder+'norm_imgs\\'+name+'_pred.png',new_img)
        return input,output, ef_pose
    


    def select_winner(self):
        
        new_list = convert_list(self.norm_list,self.ef_list)
        winner = reject_outliers(new_list,2)
        sq = 1/np.sqrt(2)
        # vec = [1,0,0]
        plot_3d_vectors(new_list,np.array(winner)*1.5)
        T10 = np.linalg.inv(self.rotation_matrix)

        self.winner_normal = winner # In 1 coordinate system
        # self.winner_normal = vec

        # 
    def save_by_the_winner(self):
        for pose in self.poses:
            filename = self.test_folder+'green_'+str(pose)+'.feather'
            load = feather.read_dataframe(filename)
            img =  np.array(load['img_color'])[0].reshape(load['shape_color'][0])
            plt.imsave('images/test/temp.png',img)
            img = cv.imread('images/test/temp.png')
            # _,green = get_green_img(img)
            ef = load['end_effector_pose']
            vec = find_vec_in_ef_system(ef,self.winner_normal)
            new_img = draw_normal_on_image(vec,img)
            
            plt.imsave(self.test_folder+str(pose)+'_pred.png',new_img)


    def calc_real_orientation(self,leaf_number):
        desired_file = self.new_test_folder+'N'+str(leaf_number)+'_d0.2_p0_r0_O0.feather'
        ind_desired_file = np.where(np.array(self.poses)==desired_file[5:])[0][0]
        joints =  self.inputs[ind_desired_file]
        if not self.no_ef:
            joints = joints[:-12]
        fix_matrix = get_fix_matrix(desired_file)
        self.real_orientation = []
        for i in range(len(self.poses)):
            normal = find_normals(filename_new_img=self.poses[i][:-8],fix_matrix=fix_matrix)
            temp_normal = np.zeros_like(normal)
            # temp_normal[0] = normal[2]
            # temp_normal[1] = normal[1]
            # temp_normal[2] = normal[0]
            # normal = temp_normal
            self.real_orientation.append(normal)

        # self.real_orientation = [[1,1,1],[1,1,1],[1,1,1],[1,1,1]]

    def calc_vec(self,save=True):
        if len(self.poses)>2:
            # combos = self.get_all_combos()
            print('Estimating Leaf normal vector..')
            self.combos_calculated = []
            # self.all_poses = []
            self.norm_list = []
            self.inputs = []
            # for combo in combos:
                # if combo not in (self.combos_calculated):
            combo = [0,1,2]
            input, output,ef_pose = self.get_vector(combo,save=save)
            self.norm_list.append(output[0])
            # self.ef_list.append(ef_pose)
            self.combos_calculated.append(combo)
            self.inputs.append(input)
            # self.all_poses.append(self.poses)
    
    def save_all_cases(self):
        
        predicted = []
        real = []
        error = []
        pose_1 = []
        pose_2 = []
        pose_3 = []
        delta_elevation_1 = []
        delta_elevation_2 = []
        delta_distance_1 = []
        delta_distance_2 = []
        leaf_number = []
        ef_pose_1 = []
        ef_pose_2 = []
        ef_pose_3 = []
        

        for i in range(len(self.combos_calculated)):
            ind_pose = self.combos_calculated[i]
            true_vec = self.real_orientation[ind_pose[2]]
            pred = fix_vecs(self.norm_list[i],true_vec)
            predicted.append(pred)
            real.append(true_vec)
            error.append(error_angle(pred,true_vec))

            


            pose_1.append(self.poses[ind_pose[0]][:-8])
            pose_2.append(self.poses[ind_pose[1]][:-8])
            pose_3.append(self.poses[ind_pose[2]][:-8])

            params = get_params(self.poses[ind_pose[0]][:-8])
            dis_1,elev_1,leaf_num = params[1],params[2],params[0]
           
            params =  get_params(self.poses[ind_pose[1]][:-8])
            dis_2,elev_2 = params[1],params[2]

            params =  get_params(self.poses[ind_pose[2]][:-8])
            dis_3,elev_3 = params[1],params[2]

            delta_elevation_1.append(elev_2-elev_1)
            delta_elevation_2.append(elev_3-elev_2)
            delta_distance_1.append(dis_2-dis_1)
            delta_distance_2.append(dis_3-dis_2)
            leaf_number.append(leaf_num)




        dataframe = {'predicted': [], 'real': [] , 'error': [], 'pose_1': [], 'pose_2': [], 'pose_3': [],
                     'delta_elevation_1': [],'delta_elevation_2': [],'delta_distance_1': [],'delta_distance_2': [],
                     'leaf_number': [],'inputs': []}
        dataframe['predicted'] = predicted
        dataframe['real'] = real
        dataframe['error'] = error
        dataframe['pose_1'] = pose_1
        dataframe['pose_2'] = pose_2
        dataframe['pose_3'] = pose_3
        dataframe['delta_elevation_1'] = delta_elevation_1
        dataframe['delta_elevation_2'] = delta_elevation_2
        dataframe['delta_distance_1'] = delta_distance_1
        dataframe['delta_distance_2'] = delta_distance_2
        dataframe['leaf_number']= leaf_number
        dataframe['inputs']= self.inputs

        return dataframe

    def covert_normal_to_robotic_pose(self):
        T01 = self.rotation_matrix
        pose = np.zeros(6)
        old_normal = self.norm_list[0] #normal in robot base coordinate systme
        
        old_normal_2 = [-old_normal[2],old_normal[1],old_normal[0]]
        old_normal_2 = [-old_normal[2],0,old_normal[0]]
        normal = old_normal_2 

        xb = 0
        try:
            zb_1 = np.sqrt((normal[1]**2)/(normal[1]**2+normal[2]**2))
            zb_2 = -zb_1
            if (np.sqrt(1-zb_1**2))*normal[1]+zb_1*normal[2]==0:
                zb = zb_1
            else:
                zb = zb_2
        except:
            zb = 0
        yb = np.sqrt(1-zb**2)
        c_vec = normal
        b_vec = [xb,yb,zb]
        a_vec = np.cross(b_vec,c_vec)

        TLtagL=np.zeros((4,4))
        TLtagL[0:3,0]=a_vec
        TLtagL[0:3,1]=b_vec
        TLtagL[0:3,2]=c_vec
        TLtagL[3,3]= 1 

        T1Ltag  = np.array([[1, 0, 0, self.leaf_gripper[0]],
                    [0, 1, 0, self.leaf_gripper[1]],
                    [0, 0, 1, self.leaf_gripper[2]],
                    [0, 0, 0, 1]]) # Move To plant coordination system

        
        TLtempGripper  = np.array([[0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]])


        T_tot = np.dot(T01,T1Ltag)
        T_tot = np.dot(T_tot,TLtagL)
        self.T_tot = np.dot(T_tot,TLtempGripper)

        self.TLtempGripper1  = np.array([[1, 0, 0.2, 0],
            [0, 1, 0, -0.08],
            [0, 0, 1, -0.33],
            [0, 0, 0, 1]]) 

        T_1 = np.dot(self.T_tot,self.TLtempGripper1)

        rot_vec = Rotation.from_matrix(T_1[0:3,0:3]).as_rotvec()
        location = T_1[:,3][0:3]
        #step_1
        pose[0:3] = location
        pose[3:6] = rot_vec
        self.robotic_step_1 = pose

        TLtempGripper2  = self.TLtempGripper1
        TLtempGripper2[2,3] = self.TLtempGripper1[2,3] + 0.3

        step_2 = np.zeros(6)
        T_2  = np.dot(self.T_tot,TLtempGripper2)
        rot_vec_2 = Rotation.from_matrix(T_2[0:3,0:3]).as_rotvec()
        location_2 = T_2[:,3][0:3]
        #step_1
        step_2[0:3] = location_2
        step_2[3:6] = rot_vec_2

        self.robotic_step_2 = step_2

    def move_and_measure(self):
        # self.covert_normal_to_robotic_pose()
        self.connect_ur5()
        self.move(self.robotic_step_1,'pose')
        time.sleep(4)
        #step_2
        self.connect_ur5()
        self.move(self.robotic_step_2,'pose')
        time.sleep(4)

