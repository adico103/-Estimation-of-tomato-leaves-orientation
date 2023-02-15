import matplotlib
from numpy.core.fromnumeric import reshape, shape
matplotlib.use("Agg")
import cv2 as cv
import pyrealsense2 as rs
import matplotlib.backends.backend_agg as agg
import pylab
import scipy.io
import os
import sys
from os.path import dirname, join as pjoin
import numpy as np
import pandas as pd
import feather

# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# rawdata = pjoin(dirname(scipy.io.__file__), currentdir, 'rawdata')

class Camera:
    def __init__(self):
        # self.fig = pylab.figure(figsize=[P_X/DPI, P_Y/DPI], # Inches
        #            dpi=DPI,        # 100 dots per inch, so the resulting buffer is 400x400 pixels
        #            )
        self.fig = pylab.figure(figsize=[5, 5], # Inches
                   dpi=100,        # 100 dots per inch, so the resulting buffer is 400x400 pixels
                   )
        self.ax = self.fig.gca()
        self.HEIGHT = 480
        self.WIDGHT = 640
        self.pipeline = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_stream(rs.stream.depth, self.WIDGHT,
                               self.HEIGHT, rs.format.z16, 30)
        self.cfg.enable_stream(rs.stream.color,
                               self.WIDGHT,
                               self.HEIGHT,
                               rs.format.rgb8, 30)
        # self.cfg.enable_stream(rs.stream.color,
        #                        self.WIDGHT,
        #                        self.HEIGHT,
        #                        rs.format.bgr8, 30)
        self.profile = self.pipeline.start(self.cfg)
        # self.play_camera()
        

    def play_camera(self):
        self.ax.cla()
        # align_to = rs.stream.color
        # align = rs.align(align_to)
        fs = self.pipeline.wait_for_frames()
        # fs = align.process(fs)
##^^
        self.data_frame = {'img_color': [], 'shape_color': [], 'img_depth': [], 'shape_depth': [] , 'leaf_orientation': [], 'leaf_length': [],'leaf_width': []}
        # self.data_frame = {'img_color': [], 'shape_color': []}
        

        self.color_frame = fs.get_color_frame()
        self.depth_frame = fs.get_depth_frame()

        self.color_image = np.asanyarray(self.color_frame.get_data())
        self.depth_image = np.asanyarray(self.depth_frame.get_data())

        self.data_frame['shape_color'].append(list(self.color_image.shape))
        self.data_frame['img_color'].append(list(self.color_image.flatten()))
        self.data_frame['shape_depth'].append(list(self.depth_image.shape))
        self.data_frame['img_depth'].append(list(self.depth_image.flatten()))

        self.data_frame['leaf_orientation'].append(list(np.array([0,0,0])))
        self.data_frame['leaf_length'].append(list(np.array([self.leaf_length])))
        self.data_frame['leaf_width'].append(list(np.array([self.leaf_width])))

        self.data_frame = pd.DataFrame(self.data_frame)


        self.data_frame.columns = self.data_frame.columns.astype(str)

        self.data_frame.reset_index(drop=True).to_feather('data.feather')
        # self.depth_frame = fs.get_depth_frame()
        # # self.depth_frame = fs.as_frameset().get_depth_frame() 
        # self.data_frame['shape_depth'].append(list(self.depth_frame.shape))
        # self.data_frame['img_depth'].append(list(self.depth_frame.flatten()))


        # if not depth_frame or not color_frame:
        #     continue


        # arrange 3d:
        # new=self.color_image[:,:,0]
        # print(new.shape)
        # self.color_image = pd.DataFrame(self.color_image)
        # self.color_image.columns = self.color_image.columns.astype(str).values

        # # Load
        self.data_load = feather.read_dataframe('data.feather')

        img_color = self.data_load['img_color'].to_numpy()[0]
        shape_color = self.data_load['shape_color'].to_numpy()[0]
        img_color = np.reshape(img_color,shape_color)

        img_depth = self.data_load['img_depth'].to_numpy()[0]
        shape_depth = self.data_load['shape_depth'].to_numpy()[0]
        # print('shape_color', self.data_load['shape_depth'])

        img_depth = np.reshape(img_depth,shape_depth)




        # # Back to numpy
        # self.data_load = self.data_load.to_numpy()
        # print('data_load',self.data_load)

        # print(np.shape(self.color_image))
        # df = pd.DataFrame(self.color_image)
        # self.depth_image = np.asanyarray(self.depth_frame.get_data())
        # print(np.shape(self.depth_image))



        # Print in black and white
        # depth_image = cv.convertScaleAbs(depth_image, alpha=0.03)
        # Print in color
        # depth_image = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.1), cv.COLORMAP_JET)
        window = cv.namedWindow('window', cv.WINDOW_AUTOSIZE)
        cv.imshow('window', img_color)
        cv.waitKey(1)
        self.ax.imshow(img_color, interpolation='bicubic')
        # self.ax.imshow(self.color_image)
        # self.ax.axis('off')
        # self.ax.set_frame_on(False)
        # self.canvas = agg.FigureCanvasAgg(self.fig)
        # self.canvas.draw()
        # self.renderer = self.canvas.get_renderer()
        # self.raw_data = self.renderer.tostring_rgb()
    
    def save_img(self, name):
        path = "C:\data_exp"
        depth_name = name + ' depth'


        # saving as csv
        # name_csv = name + '.csv'
        # depth_csv = depth_name + '.csv'
        # cv.imwrite(name_csv, self.color_image)
        # cv.imwrite(depth_csv, self.depth_image)

        # name = pjoin(rawdata, name)
        # saving as png
        name = name + '.png'
        depth_name = depth_name + '.png'
        # cv.imwrite(name, self.color_image)
        # cv.imwrite(depth_name, self.depth_image)
        
        # save_color_image = cv.cvtColor(self.color_image, cv.COLOR_RGB2BGR)
        # cv.imwrite(os.path.join(path , depth_name), self.depth_image)

camera= Camera()
camera.leaf_length = input("Enter leaf_length (mm) :")
camera.leaf_width = input("Enter leaf_width (mm) :")
while True:
    

    camera.play_camera()
    camera.save_img('adi')

