from hashlib import new
import cv2 as cv
import numpy as np
import math
from numpy import cos, empty, sin, tan, arctan2, arccos, arcsin, pi, sqrt
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import regex as re
from scipy.spatial.transform import Rotation as R
import feather
import pandas as pd

def save_df(all_data,new_name):
    all_data  = pd.DataFrame(all_data )
    all_data.columns = all_data .columns.astype(str)
    all_data.reset_index(drop=True).to_feather(new_name)

# def get_green_img(img_color):
#     # img_color = cv.cvtColor(img_color, cv.COLOR_BGR2RGB)
#     # mask green
#     # hsv = cv.cvtColor(img_color, cv.COLOR_BGR2HSV)
#     hsv = cv.cvtColor(img_color, cv.COLOR_RGB2HSV)
#     lower_hsv, upper_hsv = np.array([30, 30, 20]), np.array([90, 255, 255])
#     # lower_hsv, upper_hsv = np.array([36, 60, 40]), np.array([90, 255, 255])
#     mask_green = cv.inRange(hsv, lower_hsv, upper_hsv)
#     green = cv.bitwise_and(img_color, img_color, mask=mask_green)
    
#     return mask_green,green


import numpy as np
import cv2 as cv

buffer = 100

def draw_4_normals_on_image(preds,true,img_color,colors):

    
    mask_green,green = get_green_img(img_color)
    joints,center = get_center(green)
    imask_green = mask_green > 0
    new_green = np.zeros_like(green, np.uint8)
    new_green = new_green +255 # background color
    new_green[imask_green] = green[imask_green] 
    #pred
    slope_1,l1 = convert_normal_to_image_slope(preds[0])
    slope_2,l2 = convert_normal_to_image_slope(preds[1])
    slope_3,l3 = convert_normal_to_image_slope(preds[2])
    slope_4,l4 = convert_normal_to_image_slope(true)
    limits_pred_1 = find_limits(slope_1,center,vec_len=l1)
    limits_pred_2 = find_limits(slope_2,center,vec_len=l2)
    limits_pred_3 = find_limits(slope_3,center,vec_len=l3)
    limits_true = find_limits(slope_4,center,vec_len=l4)
    all_limits = [limits_pred_1,limits_pred_2,limits_pred_3]

    image = draw_4_arrows(center,all_limits,limits_true,new_green,colors)

    return image

def draw_normal_on_image(normal,img_color,center,vec_len=1):

    
    slope,l = convert_normal_to_image_slope(normal)
    limits_normal = find_limits(slope,center,vec_len=vec_len)
    image = draw_arrow(center,limits_normal,img_color,vec_len=vec_len)

    return image

def draw_2_normals_on_image(pred,true,img_color,crop=True,isgreen=True):

    mask_green,green = get_green_img(img_color)
    joints,center = get_center(green)
    imask_green = mask_green > 0
    new_green = np.zeros_like(green, np.uint8)
    new_green = new_green +255 # background color
    new_green[imask_green] = green[imask_green]
    if not isgreen:
        new_green = img_color
    #pred
    slope_1 ,l1= convert_normal_to_image_slope(pred)
    slope_2 ,l2 = convert_normal_to_image_slope(true)
    limits_pred = find_limits(slope_1,center,vec_len=l1)
    limits_true = find_limits(slope_2,center,vec_len=l2)

    image = draw_arrows(center,limits_pred,limits_true,new_green)
    if crop:
        image = crop(image,center,buffer)

    return image

def draw_arrow(center,limits_pred,green,vec_len=1):
    red = (255 ,0 ,0)
    blue = (0 ,0 ,255)
    green = cv.arrowedLine(green, [int(center[0]),int(center[1])],[limits_pred[0],limits_pred[1]] ,color=red,thickness=1)
    return green


def draw_arrows(center,limits_pred,limits_true,green):
    red = (255 ,0 ,0)
    blue = (0 ,0 ,255)
    
    green = cv.arrowedLine(green, [int(center[0]),int(center[1])], [limits_pred[0],limits_pred[1]],color=red,thickness=1)
    green = cv.arrowedLine(green, [int(center[0]),int(center[1])], [limits_true[0],limits_true[1]],color=blue,thickness=2)
    return green

def draw_4_arrows(center,limits_pred,limits_true,green,colors):
    
    for limits,color in zip(limits_pred,colors[:3]):
        green = cv.arrowedLine(green, [int(center[0]),int(center[1])], [limits[0],limits[1]],color=color,thickness=1)

    
    green = cv.arrowedLine(green, [int(center[0]),int(center[1])], [limits_true[0],limits_true[1]],color=color[-1],thickness=2)
    return green


def get_green_img(img_color,lower_hsv=np.array([34, 40, 30]),upper_hsv= np.array([100, 255, 255])):

    temp_im = img_color
    img_color = cv.cvtColor(img_color, cv.COLOR_BGR2RGB)
    # mask green
    hsv = cv.cvtColor(img_color, cv.COLOR_BGR2HSV)
    # lower_hsv, upper_hsv =np.array([34, 50, 40]), np.array([90, 255, 255]))
    mask_green = cv.inRange(hsv, lower_hsv, upper_hsv)
    green = cv.bitwise_and(img_color, img_color, mask=mask_green)
    
    return mask_green,green

def draw_contours(img_filter):
    ''' Draw Contours '''
    img_gray = cv.cvtColor(img_filter, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(img_gray, 5, 255, cv.THRESH_BINARY)

    try:
        contours, _ = cv.findContours(image=thresh, mode=cv.RETR_CCOMP, method=cv.CHAIN_APPROX_SIMPLE)
    except ValueError:
        _, contours, _ = cv.findContours(image=thresh, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)

    idx = 0
    contours_lenght = 0

    for i, c in enumerate(contours):
        if contours_lenght < len(c):
            contours_lenght = len(c)
            idx = i
    
    choose_contours = contours[idx]

    cv.drawContours(image=img_filter, contours=[choose_contours], contourIdx=-1, color=(0, 255, 0), thickness=1, lineType=cv.LINE_AA)

    return img_filter, choose_contours

def convert_normal_to_image_slope(normal):
    x, y,z = normal
    x_image , y_image = -z,x
    if x_image == 0.0:
        length = 200*(abs(1-y))
        return 10000000000, length
    slope = y_image/x_image
    length = 200*(abs(1-y))
    
    return slope,length

def get_joints(contours, n=16,con=False):
    
    joints = []

    for idx in np.linspace(0, len(contours)-1, n).astype(int):
        joints.append(contours[idx][0])
    if con:
        return np.array(joints),contours.flatten().reshape(-1,2)

    return np.array(joints)


def draw_circles(img, joints):
    joints = joints.flatten().reshape(-1,2)
    for joint in joints:
        joint = [int(joint[0]),int(joint[1])]
        cv.circle(img, tuple(joint), 2, (0, 0, 255), 2)

    return img


def get_center(green,con=False):
    try:
        _,contours = draw_contours(green)
        if con:
            joints,contour  = get_joints(contours, n=20,con=con)
        else:
            joints  = get_joints(contours, n=20,con=con)
            
        x = joints.flatten()[0::2]
        y = joints.flatten()[1::2]
        center = np.mean(x),np.mean(y)
    except:
        joints = np.array([[0,0],[0,0]])
        center = [0,0]
    if con:
        return joints,center,contour
    return joints,center


def find_limits(slope,center,vec_len = 100):
    if slope>0:
        vec_len = -vec_len
    x_center,y_center = center
    b = y_center-slope*x_center
    temp_x = 200
    temp_y = slope*temp_x + b
    dis = np.sqrt((center[0]-temp_x)**2+(center[1]-temp_y)**2)
    end_x = center[0] + vec_len*(center[0]-temp_x)/(dis)
    end_y = slope*end_x + b

    return int(end_x), int(end_y)

def get_limits(center ,buffer, lim_x, lim_y):

    x_34 , x12 , y23 , y14 = center[0]+buffer , center[0]-buffer , center[1]+buffer ,center[1]-buffer
    check = np.array([lim_x-x_34,x12,lim_y-y23,y14])
    check[np.where(check>0)]=0
    new_center = center[0]+check[0]-check[1] , center[1]+check[2]-check[3]
    return new_center

def crop(green , center ,buffer):

    lim_x = np.shape(green)[0]
    lim_y = np.shape(green)[1]
    # center = get_limits(center ,buffer,lim_x, lim_y)
    x_34 , x12 , y23 , y14 = int(center[0]+buffer) , int(center[0]-buffer) , int(center[1]+buffer) ,int(center[1]-buffer)
    green = green[x12:x_34,y14:y23]
    return green

def draw_on_image(green ,normal_mat, name= 'test_image', color = 'red'):
    if (color == 'red'):
        green[normal_mat] = [255 ,0 ,0]
    if (color == 'blue'):
        green[normal_mat] = [0 ,0 ,255]
    if (color == 'green'):
        green[normal_mat] = [0 ,255 ,0]
    name = name + '.png'
    cv.imwrite(name, green)

def draw_2_on_image(green ,normal_mat, second_mat):
    green[normal_mat] = [255 ,0 ,0]
    green[second_mat] = [0 ,0 ,255]
    return green

n_joints = 20

def get_geometry(joints):
        joints = joints[0:2*n_joints]
        x = joints[:,0]
        y = joints[:,1]
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

def get_leaf_geo(joints):
        min_x,min_y,max_x,max_y,long_dis,long_slope,short_dis,short_slope = get_geometry(joints)  
        del_x = max_x-min_x
        del_y = max_y-min_y
        x_y_ratio = del_y/del_x
        main_ratio = long_dis/short_dis
        return x_y_ratio,main_ratio,long_slope

def get_leaf_surface(joints):
    _,_,_,_,long_dis,_,short_dis,_ = get_geometry(joints)  
    sur = long_dis*short_dis
    return sur

def get_tcp_wanted_rot(rotation_matrix, d, th_tcp, r_plant,th_rot):
    # initiate plant position
    p = np.zeros([4, 4])
    T_01 = rot_around_z(th_rot)
    T_01 = rotation_matrix
    T_1p = np.array([[1, 0, 0, d[0]],
                    [0, 1, 0, d[1]],
                    [0, 0, 1, d[2]],
                    [0, 0, 0, 1]]) # Move To plant coordination system
    T_p2 = np.array([[1, 0, 0, 0],
                    [0, -sin(-th_tcp), cos(-th_tcp), 0],
                    [0, -cos(-th_tcp), -sin(-th_tcp), 0],
                    [0, 0, 0, 1]]) # Rotation around X with -(90+th_tcp)
    T_23 = np.array([[0, 1, 0, 0],
                    [-1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]]) # Changing axis names
    T_3c =  np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, -r_plant],
                    [0, 0, 0, 1]]) # Move To plant coordination system
    loc = []
    # T = np.dot(T_minus10, T_01)
    T = np.dot(T_01, T_1p)
    T = np.dot(T, T_p2)
    T = np.dot(T, T_23)
    T = np.dot(T, T_3c)
    return T

def rot_around_z(th_rot):
    T_01 = np.array([[cos(th_rot), -sin(th_rot), 0, 0],
            [sin(th_rot), cos(th_rot), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])  # Rotation of the base around Z axis by th_rot
    return T_01

def rot_around_x(th_rot):
    T = np.array([[1, 0, 0, 0],
                    [0, cos(th_rot), -sin(th_rot), 0],
                    [0, sin(th_rot), cos(th_rot), 0],
                    [0, 0, 0, 1]]) # Rotation around X with -(90+th_tcp)
    return T

def rot_around_y(th_rot):
    T = np.array([[cos(th_rot), 0, sin(th_rot), 0],
                    [0, 1, 0, 0],
                    [-sin(th_rot), 0, cos(th_rot), 0],
                    [0, 0, 0, 1]]) # Rotation around X with -(90+th_tcp)
    return T


def get_rotational_path(r_plant,d_base,rotation_matrix,th_rot, move_type='pose',angles = []):
        if move_type == 'pose':
            p = np.empty([0, 6])
            th_tcp_vec = best_route_geometry(angles)
            for th_tcp in th_tcp_vec:
                T_temp = get_tcp_wanted_rot(rotation_matrix, d_base, th_tcp, r_plant,th_rot)
                p_temp = T_temp[:3, 3]
                rv_temp = Rotation.from_matrix(T_temp[:3, :3])
                rv_temp = rv_temp.as_rotvec()
                p = np.vstack([p, np.hstack([p_temp[:], rv_temp[:]])])
            return p,np.round(th_tcp_vec*180/np.pi)

def best_route_geometry(angles):
    th_tcp_vec = np.array(angles).astype(float)*np.pi/180
    return th_tcp_vec

def define_geometry(r_plant, th_tcp_steps, th_tcp_range):
    if isinstance(th_tcp_steps, type(None)) is False:
        th_tcp_vec = np.linspace(th_tcp_range[0], th_tcp_range[1], th_tcp_steps)*np.pi/180
        return th_tcp_vec

def find_angle(vector_1,vector_2):
    unit_vector_1 = vector_1 / np. linalg. norm(vector_1)
    unit_vector_2 = vector_2 / np. linalg. norm(vector_2)
    dot_product = np. dot(unit_vector_1, unit_vector_2)
    return np. arccos(dot_product)

def norm_pixels(img_joints,min_x=0,min_y=0,first_im=True):
    img_joints=img_joints.flatten()
    x_pixels = img_joints[0::2]
    y_pixels = img_joints[1::2]
    if first_im:
        min_x = np.min(x_pixels)
        min_y = np.min(y_pixels)

    x_pixels = x_pixels-min_x
    y_pixels = y_pixels-min_y

    new_joints = [item for pair in zip(x_pixels, y_pixels) for item in pair]
    if first_im:
        return min_x,min_y,np.array(new_joints).flatten() 

    return np.array(new_joints).flatten()  

def reject_outliers(vecs, m=2):
    vecs = np.array(vecs)
    angles = []
    mean_vec =  np.mean(vecs[:,0]),np.mean(vecs[:,1]),np.mean(vecs[:,2])
    for i in range(len(vecs)):
        angles.append(find_angle(mean_vec,vecs[i]))
    angles = np.array(angles)
    new_vecs = vecs[abs(angles) < m * np.std(angles)]
    out = np.mean(vecs[:,0]),np.mean(vecs[:,1]),np.mean(vecs[:,2])
    if not out is empty:
        return  np.mean(vecs[:,0]),np.mean(vecs[:,1]),np.mean(vecs[:,2])
    return  np.mean(new_vecs[:,0]),np.mean(new_vecs[:,1]),np.mean(new_vecs[:,2])


def conv_vec(vec1,ef_1,ef_2): #Getting vec1 to coor sys of ef_2
    # Rotation.from_rotvec
    rot_1 = ef_1[3:]
    rot_2 = ef_2[3:]
    T01 = Rotation.from_rotvec(rot_1).as_matrix()
    T02 = Rotation.from_rotvec(rot_2).as_matrix()
    T20 = np.linalg.inv(T02)
    T21 = np.dot(T20,T01)
    return np.dot(T21,vec1)

def convert_list(norm_list,ef_list):
    new_list = []
    
    for i in range(len(norm_list)):
        win_norm = norm_list[i]
        win_ef_pose = ef_list[i][0]
        rot = win_ef_pose[3:]
        vec = np.zeros(3)
        for i in range(len(vec)):
            vec[i] = rot[i]
        T0c = Rotation.from_rotvec(vec).as_matrix()
        new_norm = np.dot(T0c,win_norm)
        new_list.append(new_norm)

    return new_list

def plot_3d_vectors(new_list,winner):
    fig = plt.figure()
    arr = np.array(new_list)
    colors = ['b','g','r','c','m','y']
    ax = fig.add_subplot(111, projection='3d')
    origin = [0,0,0]
    ind = 0
    for i in range(len(new_list)):
        X, Y ,Z = zip(origin)
        U,V,W = zip(arr[i])
        ax.quiver(X,Y,Z,U,V,W,arrow_length_ratio=0.2, color=colors[ind])
        ind+=1
        if ind==6:
            ind = 0
    X, Y ,Z = zip(origin)
    U,V,W = zip(winner)
    ax.quiver(X,Y,Z,U,V,W,arrow_length_ratio=0.5, color='k')
    

    ax.set_xlim([-1, 0.5])
    ax.set_ylim([-1, 1.5])
    ax.set_zlim([-1, 8])
    plt.savefig('VECTOTRS')

def find_vec_in_ef_system(ef,winner_normal):
    rot = ef[0][3:]
    vec = np.zeros(3)
    for i in range(len(vec)):
        vec[i] = rot[i]
    T0c = Rotation.from_rotvec(vec).as_matrix()
    Tc0 = np.linalg.inv(T0c)
    return np.dot(Tc0,winner_normal)

def error_angle(pred,true):
    unit_vector_1 = pred / np.linalg.norm(pred)
    unit_vector_2 = true / np.linalg.norm(true)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)*180/np.pi
    return angle

def fix_vecs(pred,true):
    angle = error_angle(pred,true)
    if (180-abs(angle))<angle:
        return -pred
    return pred

def get_params(filename,old=False,png=True):
    pattern = "N(.*?)_d"
    leaf_number = re.search(pattern, filename).group(1)
    pattern = "d(.*?)_p"
    distance = re.search(pattern, filename).group(1)
    pattern = "p(.*?)_r"
    elevation = re.search(pattern, filename).group(1)
    pattern = "r(.*?)_O"
    rotation = re.search(pattern, filename).group(1)
    if not old:
        pattern = "O(.*)"
        orientation = re.search(pattern, filename).group(1)
    else:
        if png:
            pattern = "O(.*?)new"
            orientation = re.search(pattern, filename).group(1)
        else:
            pattern =  "O(.*)"
            orientation = re.search(pattern, filename).group(1)
    parameters = np.float16([leaf_number,distance,elevation,rotation,orientation])
    return parameters


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

def get_boundry(load,old=False):

    fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
    
    lower_hsv=np.array([34, 40, 30])
    upper_hsv= np.array([100, 255, 255])
    if old:
        lower_hsv=np.array([34, 45, 40])
        upper_hsv= np.array([90, 255, 255])
    img =  np.array(load['img_color'])[0].reshape(load['shape_color'][0])
    # ax1.imshow(img)
    
    
    plt.imsave('images/test/temp.png',img)
    img = cv.imread('images/test/temp.png')
    # _,green = get_green_img(img)
    _,green = get_green_img(img,lower_hsv=lower_hsv,upper_hsv=upper_hsv)
    
    p = green
    p[0:250,0:680] = [0,0,0]
    p[0:480,600:680] = [0,0,0]
    # d=0
    # ax2.imshow(p)
    joints, center ,contur= get_center(p,con=True)
    # t_m = draw_circles(p,joints=joints)
    # ax3.imshow(t_m)
    # joints, center ,contur= get_center(green,con=True)

    return joints, green,contur


def get_fix_matrix(filename,from_file=True,joints=[]):
    if from_file:
        load = feather.read_dataframe(filename)
        # b = np.array(load['joints'][0])
        b,_,_ =get_boundry(load)
    else:
        b= joints
    b= b.flatten()
    x= b[0::2]
    y= b[1::2]
    m, b = np.polyfit(y, x, 1)
    # m, b = np.polyfit(x, y, 1)
    fix_angle= np.pi/2+ np.arctan(m)
    fix_matrix = np.array([[1, 0, 0],
    [0, cos(-fix_angle), -sin(-fix_angle)],
    [0, sin(-fix_angle), cos(-fix_angle)]])   

    return fix_matrix

def get_perfect_rot_matrix(T0leaf, fix_matrix):
    new_mat = np.dot(T0leaf,fix_matrix)
    normal = np.dot(new_mat,[0,0,1])
    return new_mat, normal


def find_normals(filename_new_img,fix_matrix,old=False,png=True):

    leaf_number,distance,elevation,rotation,orientation = get_params(filename_new_img,old=old,png=png)
    T0leaf = calc_rotation_mat(elevation,rotation,orientation)
    
    _ ,normal = get_perfect_rot_matrix(T0leaf, fix_matrix)

    return normal

def change_direction_joints(joints):
    joints = joints.flatten().reshape(-1,2)
    fixd =  np.zeros_like(joints)
    inds = np.arange(0,len(joints),1)
    ind_min_y = np.argmin(joints[:,1])
    fixd[inds] = joints[inds-(19-ind_min_y)]
    fixd = fixd[::-1]
    return fixd.flatten()