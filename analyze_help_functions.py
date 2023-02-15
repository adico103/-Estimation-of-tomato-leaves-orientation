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
    slope_1 = convert_normal_to_image_slope(preds[0])
    slope_2 = convert_normal_to_image_slope(preds[1])
    slope_3 = convert_normal_to_image_slope(preds[2])
    slope_4 = convert_normal_to_image_slope(true)
    limits_pred_1 = find_limits(slope_1,center,vec_len=150)
    limits_pred_2 = find_limits(slope_2,center,vec_len=130)
    limits_pred_3 = find_limits(slope_3,center,vec_len=110)
    limits_true = find_limits(slope_4,center,vec_len=200)
    all_limits = [limits_pred_1,limits_pred_2,limits_pred_3]

    image = draw_4_arrows(center,all_limits,limits_true,new_green,colors)
    # image = draw_circles(image,joints)
    # image = crop(image,center,buffer)

    return image

def draw_1_normal_on_image(true,img_color):

    
    mask_green,green = get_green_img(img_color)
    joints,center = get_center(green)
    imask_green = mask_green > 0
    new_green = np.zeros_like(green, np.uint8)
    new_green = new_green +255 # background color
    new_green[imask_green] = green[imask_green] 
    #pred
    slope_2 = convert_normal_to_image_slope(true)
    limits_true = find_limits(slope_2,center,vec_len=150)

    image = draw_arrows(center,limits_true=limits_true,green = new_green,two_arr=False)
    # image = draw_circles(image,joints)
    image = crop(image,center,buffer)

    return image



def draw_2_normals_on_image(pred,true,img_color):

    
    mask_green,green = get_green_img(img_color)
    joints,center = get_center(green)
    imask_green = mask_green > 0
    new_green = np.zeros_like(green, np.uint8)
    new_green = new_green +255 # background color
    new_green[imask_green] = green[imask_green] 
    #pred
    slope_1 = convert_normal_to_image_slope(pred)
    slope_2 = convert_normal_to_image_slope(true)
    limits_pred = find_limits(slope_1,center,vec_len=100)
    limits_true = find_limits(slope_2,center,vec_len=150)

    image = draw_arrows(center,limits_pred,limits_true,new_green)
    # image = draw_circles(image,joints)
    image = crop(image,center,buffer)

    return image

def draw_arrows(center,limits_pred=[],limits_true=[],green=[],two_arr=True):
    red = (255 ,0 ,0)
    blue = (0 ,0 ,255)
    if two_arr:
        green = cv.arrowedLine(green, [int(center[0]),int(center[1])], [limits_pred[0],limits_pred[1]],color=red,thickness=1)
        green = cv.arrowedLine(green, [int(center[0]),int(center[1])], [limits_true[0],limits_true[1]],color=blue,thickness=2)
    else:
        green = cv.arrowedLine(green, [int(center[0]),int(center[1])], [limits_true[0],limits_true[1]],color=blue,thickness=2)
    return green

def draw_4_arrows(center,limits_pred,limits_true,green,colors):
    
    for limits,color in zip(limits_pred,colors[:3]):
        green = cv.arrowedLine(green, [int(center[0]),int(center[1])], [limits[0],limits[1]],color=color,thickness=1)

    
    green = cv.arrowedLine(green, [int(center[0]),int(center[1])], [limits_true[0],limits_true[1]],color=colors[-1],thickness=2)
    return green


def get_green_img(img_color):
    img_color = cv.cvtColor(img_color, cv.COLOR_BGR2RGB)
    # mask green
    hsv = cv.cvtColor(img_color, cv.COLOR_BGR2HSV)
    lower_hsv, upper_hsv = np.array([36, 40, 40]), np.array([90, 255, 255])
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

def convert_normal_to_image_slope(normal):
    x,y,z = normal
    x_image , y_image = -z, x
    slope = y_image/x_image
    if x_image == 0.0:
        return 10000000000
    return slope

def get_joints(contours, n=16):
    
    joints = []

    for idx in np.linspace(0, len(contours)-1, n).astype(int):
        joints.append(contours[idx][0])
    
    return np.array(joints)

def draw_circles(img, joints):
    imgwhite = np.zeros([480,640,3])
    h = len(imgwhite)
    w = len(imgwhite[0])

    for y in range(h):
        for x in range(w):
            imgwhite[y,x] = [255,255,255]

    for joint in joints:
        cv.circle(img, tuple(joint), 2, (255, 0, 0), 2)
    
    for joint in joints:
        cv.circle(imgwhite, tuple(joint), 2, (255, 0, 0), 2)

    return img , imgwhite


def get_center(green):
    _,contours = draw_contours(green)
    joints  = get_joints(contours, n=20)
    x = joints.flatten()[0::2]
    y = joints.flatten()[1::2]
    center = np.mean(x),np.mean(y)
    return joints,center


def find_limits(slope,center,vec_len = 100):

    x_center,y_center = center
    b = y_center-slope*x_center
    temp_x = 200
    temp_y = slope*temp_x + b
    dis = np.sqrt((center[0]-temp_x)**2+(center[1]-temp_y)**2)
    end_x = center[0] - vec_len*(center[0]-temp_x)/(dis)
    end_y = slope*end_x + b

    return int(end_x), int(end_y)

def get_limits(center ,buffer, lim_x, lim_y):
    # print('center', center)

    x_34 , x12 , y23 , y14 = center[0]+buffer , center[0]-buffer , center[1]+buffer ,center[1]-buffer
    check = np.array([lim_x-x_34,x12,lim_y-y23,y14])
    # fix_check = np.array([lim_x,0,lim_y,0])
    check[np.where(check>0)]=0
    # print(check)
    # print(check[1])
    new_center = center[0]+check[0]-check[1] , center[1]+check[2]-check[3]
    # print(new_center)
    # return center
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