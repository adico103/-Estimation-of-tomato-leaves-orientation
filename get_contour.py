from ntpath import join
import numpy as np
import cv2 as cv
import feather
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from os.path import dirname, join as pjoin


def depth_filter(img, depth, z):

    mask = np.where(depth > z[1])
    img[mask] = 0

    mask = np.where(depth < z[0])
    img[mask] = 0

    return img

def color_filter(img, lower, upper):
    ''' Color Filter '''
    # Convert BGR to HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # mask green color
    lower_red = lower
    upper_red = upper
    mask = cv.inRange(hsv, lower_red, upper_red)

    # set my output img to zero everywhere except my mask
    # img_filter = img_color.copy()
    # img_filter[np.where(mask == 0)] = 0
    res = cv.bitwise_and(img, img, mask=mask)
    
    return res

def draw_contours(img_filter):
    ''' Draw Contours '''
    img_gray = cv.cvtColor(img_filter, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(img_gray, 5, 255, cv.THRESH_BINARY)

    try:
        contours, _ = cv.findContours(image=thresh, mode=cv.RETR_CCOMP, method=cv.CHAIN_APPROX_SIMPLE)
    except ValueError:
        _, contours, _ = cv.findContours(image=thresh, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
        # contours = cv.findContours(image=thresh, mode=cv.RETR_CCOMP, method=cv.CHAIN_APPROX_SIMPLE)

    idx = 0
    contours_lenght = 0

    for i, c in enumerate(contours):
        # print('icon',i)
        # print('ccon',c)
        if contours_lenght < len(c):
            contours_lenght = len(c)
            idx = i
    # print('n',idx)
    # print('c',contours)
    choose_contours = contours[idx]

    cv.drawContours(image=img_filter, contours=[choose_contours], contourIdx=-1, color=(0, 255, 0), thickness=1, lineType=cv.LINE_AA)

    return img_filter, choose_contours

def get_points(img):
    img_gray = cv.cvtColor(img_filter, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(img_gray, 5, 255, cv.THRESH_BINARY)
    return thresh

def get_joints(contours, n=16):
    
    joints = []

    for idx in np.linspace(0, len(contours)-1, n).astype(int):
        joints.append(contours[idx][0])
    
    return np.array(joints)

def draw_circles(img, joints):

    for joint in joints:
        cv.circle(img, tuple(joint), 2, (0, 0, 255), 2)

    return img

def outliersFilter(signal, N_med=9):

    # Pad signal
    new_signal = np.zeros_like(signal)
    padded_signal = np.append(np.repeat(signal[0], N_med//2), signal)
    padded_signal = np.append(padded_signal, np.repeat(signal[-1], N_med//2))

    # Build signal based on median filter
    for i in range(len(signal)):
        sorted_short_signal = np.sort(padded_signal[i:i+N_med])
        new_signal[i] = sorted_short_signal[N_med//2]
    
    good_values = np.argwhere(np.abs(new_signal.astype(int) - signal.astype(int)) < 100).flatten()
    # print(outliers)

    return new_signal[good_values], good_values

def get_xyz(joints, img_depth):

    # Find depth
    depths = img_depth[joints[:, 1], joints[:, 0]]

    # Filter depths outlires
    depths_new, good_values = outliersFilter(depths)
    joints_new = joints[good_values]

    # Normelize xyz
    min_xy = np.min(joints_new, axis=0)
    joints_copy_xy = joints_new.copy() - min_xy
    max_xy = np.max(joints_copy_xy, axis=0)
    joints_xy_normelized = joints_copy_xy / max_xy

    depths_new = depths_new - np.min(depths_new)
    joints_z_normelized = depths_new / np.max(depths_new)

    final_joints = np.array([joints_xy_normelized[:, 1], joints_xy_normelized[:, 0], joints_z_normelized]).T
    
    return final_joints

def get_contour(n,image, depth):
    # Reaf feather
    # image_file = feather.read_dataframe('N1_d0.3_p0.0_r0_O0.feather').iloc[0]

    # RGB image
    # image = image_file['img_color'].reshape(image_file['shape_color'])
    # image = image.swapaxes(0, 1)

    # Depth image
    # depth = image_file['img_depth'].reshape(image_file['shape_depth'])
    # depth = depth.swapaxes(0, 1)

    lower_hsv, upper_hsv = np.array([36, 40, 40]), np.array([90, 255, 255])
    filter_image = color_filter(image.copy(), lower_hsv, upper_hsv)
    filter_image, contours = draw_contours(filter_image)
    try:
        joints = get_joints(contours, n)
        return joints
        # filter_image = draw_circles(filter_image, joints)
    except TypeError:
        pass
    

    # cv.imshow('Image', filter_image)

    # cv.waitKey(0)  # waits until a key is pressed
    # cv.destroyAllWindows()  # destroys the window showing image

def get_all_joints(file,N):
    
    count = 0
    new_file = file
    new_file = new_file.drop(['img_color', 'shape_color','img_depth','shape_depth'], axis=1)
    new_file['joints'] = ""
    for n in N:
        name = str(n)+'_joints'
        new_file[name] = ""
        for i in range(len(file)):
            # print(i)
            # print(file.iloc[i])
            image = file['img_color'][i].reshape(file['shape_color'][i])
            image = image.swapaxes(0, 1)
            depth = file['img_depth'][i].reshape(file['shape_depth'][i])
            depth = depth.swapaxes(0, 1)
            new_joints = get_contour(n,image,depth)
            # print(new_joints)
            new_file[name][i] = new_joints.flatten()
            new_file['joints'][i] = new_joints.flatten()
            # print(new_file['joints'][i])
            # count = count + 1
    # print(new_file.columns)
    return new_file

def save_all_joints(directory,N,joints_folder):
    for file in os.listdir(directory) :
        if file.endswith('.feather'):
            # number = file[8:-8]
            feather_path = os.path.join(directory, file)
            load = feather.read_dataframe(feather_path)
            number = int(load['leaf_number'][0])
            print('finding pixels of feather file of leaf number: ', number )
            
            new_file = get_all_joints(load,N)
            print('saving...')
            data_frame = pd.DataFrame(new_file)
            data_frame.columns = data_frame.columns.astype(str)
            name =  os.path.join(joints_folder, 'joints_'+ str(number) +'.feather')
            # print(data_frame['12_joints'][0])
            data_frame.reset_index(drop=True)
            data_frame.to_feather(name)
            print('completed')      





