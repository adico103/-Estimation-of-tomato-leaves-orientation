# pyright: reportMissingImports=false, reportUndefinedVariable=false
from gui.settings import P_X, P_Y, DPI
import pygame as pg
from settings import *
import numpy as np
import scipy
import inspect
import pandas as pd
import feather

class micsGroup():
    def __init__(self):
        self.mics = []
        self.current = 0
        self.callback = None
        self.enabled = False

    def add(self, label):
        self.mics.append(label)

    def draw(self, screen):
        for mic in self.mics:
            screen.blit(mic.image, mic.rect)

    def on_click(self, pos, callback):
        self.callback = callback
        for i, mic in enumerate(self.mics):
            # mic.on_click(pos, button_name='mics', text=text)
            if mic.on_click(pos) and (mic.clicked == False):
                mic.change_image('mic_on.png')
                mic.clicked = True
                self.current = i
                self.callback()
            else:
                mic.change_image('mic_off.png')
                mic.clicked = False

    def change_image(self, event):
        for mic in self.mics:
            if mic.clicked == True:
                mic.change_text(event.unicode, event.key)

class labelsGroup():
    def __init__(self):
        self.labels = []

    def add(self, label):
        self.labels.append(label)
    
    def remove(self):
        # del self.labels
        self.labels = []

    def draw(self, screen):
        for label in self.labels:
            label.draw(screen)
    
    def on_click(self, pos):
        for label in self.labels:
            label.on_click(pos)
    
    def change_text(self, event):
        for label in self.labels:
            if label.active == True:
                label.change_text(event.unicode, event.key)
    
    def check_text(self):
        for label in self.labels:
            if label.user_text == '':
                return False
        return True

class Windows:
    def __init__(self):
        self.windows = []
    
    def add(self, window):
        self.windows.append(window)
    
    def play(self):
        pass
        
class Button(pg.sprite.Sprite):
    def __init__(self, image, position, callback=None):
        pg.sprite.Sprite.__init__(self)
        self.image = pg.image.load('gui/img/' + image)
        self.rect = self.image.get_rect(topleft=position)
        self.callback = callback
        self.clicked_enabled = True
        self.clicked = False

    def on_click(self, pos, button_name=None, text=None, button=None):
        if self.rect.collidepoint(pos) and self.clicked_enabled:
            if self.callback == None:
                return True
            self.callback(button_name, text, button)
    
    def change_image(self, image):
        self.image = pg.image.load("GUI/img/" + image)

class Label(pg.sprite.Sprite):
    def __init__(self, name, pos, init_text='', dim='', x_dist=60):
        # User text
        self.base_font = pg.font.Font('gui/KIN668.TTF', 14)
        self.user_text = init_text
        self.text_surface = self.base_font.render(self.user_text, True, (255, 255, 255))
        self.input_rect = pg.Rect(pos)
        self.color = {True: pg.Color('lightskyblue3'), False: pg.Color('gray15')}
        self.active = False

        # Label
        self.name_font = pg.font.Font('gui/KIN668.TTF', 14)
        self.label_name = self.name_font.render(name, True, (255, 255, 255))
        self.text_rect = self.label_name.get_rect()
        self.text_rect.topleft = (pos[0] - x_dist, pos[1] + 7)

        # Dimention
        self.dim_font = pg.font.Font('gui/KIN668.TTF', 14)
        self.dim_name = self.dim_font.render(dim, True, (255, 255, 255))
        self.dim_rect = self.dim_name.get_rect()
        self.dim_rect.topleft = (pos[0] + pos[2] + 5, pos[1] + 7)

    def draw(self, screen):
        pg.draw.rect(screen, self.color[self.active], self.input_rect, 2)
        screen.blit(self.text_surface, (self.input_rect.x + 5, self.input_rect.y + 7))
        screen.blit(self.label_name, self.text_rect)
        screen.blit(self.dim_name, self.dim_rect)

    def change_text(self, text, key):
        if self.active == True:
            if key == pg.K_BACKSPACE:
                self.user_text = self.user_text[:-1]
            else:
                self.user_text += text
            self.text_surface = self.base_font.render(self.user_text, True, (255, 255, 255))
    
    def on_click(self, pos):
        if self.input_rect.collidepoint(pos):
            self.active = True
        else:
            self.active = False


import matplotlib
matplotlib.use("Agg")
import cv2 as cv
import pyrealsense2 as rs
import matplotlib.backends.backend_agg as agg
import pylab
import scipy.io
import os
import sys
from os.path import dirname, join as pjoin
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
rawdata = pjoin(dirname(scipy.io.__file__), currentdir, 'rawdata')


class Camera:
    def __init__(self):
        
        self.fig = pylab.figure(figsize=[P_X/DPI, P_Y/DPI], # Inches
                   dpi=DPI,        # 100 dots per inch, so the resulting buffer is 400x400 pixels
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
        self.play_camera()

    def play_camera(self):
        self.ax.cla()
        # align_to = rs.stream.color
        # align = rs.align(align_to)
        fs = self.pipeline.wait_for_frames()
        # fs = align.process(fs)

        self.color_frame = fs.get_color_frame()
        self.depth_frame = fs.get_depth_frame()

        self.color_image = np.asanyarray(self.color_frame.get_data())
        self.depth_image = np.asanyarray(self.depth_frame.get_data())





# Prev Code
        # self.color_frame = fs.get_color_frame()
        # # self.depth_frame = fs.get_depth_frame()
        # self.depth_frame = fs.as_frameset().get_depth_frame() 


        # # if not depth_frame or not color_frame:
        # #     continue

        # self.color_image = np.asanyarray(self.color_frame.get_data())
        # self.depth_image = np.asanyarray(self.depth_frame.get_data())



        # Print in black and white
        # depth_image = cv.convertScaleAbs(depth_image, alpha=0.03)
        # Print in color
        # depth_image = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.1), cv.COLORMAP_JET)

        # window = cv.namedWindow('window', cv.WINDOW_AUTOSIZE)
        # cv.imshow('window', color_image)
        cv.waitKey(1)
        # self.ax.imshow(color_image, interpolation='bicubic')
        self.ax.imshow(self.color_image)
        self.ax.axis('off')
        # self.ax.set_frame_on(False)
        self.canvas = agg.FigureCanvasAgg(self.fig)
        self.canvas.draw()
        self.renderer = self.canvas.get_renderer()
        self.raw_data = self.renderer.tostring_rgb()
    
    def save_img(self, name ,leaf_length,leaf_width,leaf_orientation_matrix, leaf_origin ,leaf_orientation_vector ,end_effector_pose):
        self.data_frame = {'img_color': [], 'shape_color': [], 'img_depth': [], 'shape_depth': [] ,'leaf_origin': [], 'leaf_orientation_matrix': [], 'leaf_orientation_vector': [], 'leaf_length': [],'leaf_width': [], 'end_effector_pose': []}
        self.data_frame['shape_color'].append(list(self.color_image.shape))
        self.data_frame['img_color'].append(list(self.color_image.flatten()))
        self.data_frame['shape_depth'].append(list(self.depth_image.shape))
        self.data_frame['img_depth'].append(list(self.depth_image.flatten()))
        self.data_frame['leaf_orientation_matrix'].append(list(np.asanyarray(leaf_orientation_matrix).flatten()))
        self.data_frame['leaf_orientation_vector'].append(list([leaf_orientation_vector]))
        self.data_frame['leaf_origin'].append(list([leaf_origin]))
        self.data_frame['leaf_length'].append(list(np.array([leaf_length])))
        self.data_frame['leaf_width'].append(list(np.array([leaf_width])))
        self.data_frame['end_effector_pose'].append(list([end_effector_pose]))


        self.data_frame = pd.DataFrame(self.data_frame)
        self.data_frame.columns = self.data_frame.columns.astype(str)
        name = "C:\data_exp\{}".format(name)
        name = name + '.feather'

        self.data_frame.reset_index(drop=True).to_feather(name)
        
class Graph:
    def __init__(self):
        self.fig = pylab.figure(figsize=[4, 2.6],  # Inches
                                dpi=100,        # 100 dots per inch, so the resulting buffer is 400x400 pixels
                                )
        self.ax = self.fig.gca()
        self.current = False
        self.enabled = False
        self.t = None
        self.fs = None
        self.x = None
        # self.play_camera()

    def render(self):
        self.canvas = agg.FigureCanvasAgg(self.fig)
        self.canvas.draw()
        self.renderer = self.canvas.get_renderer()
        self.raw_data = self.renderer.tostring_rgb()

        self.size = self.canvas.get_width_height()
        self.surf = pg.image.fromstring(self.raw_data, self.size, "RGB")

    def plot_time(self):
        self.ax.cla()
        self.ax.plot(self.t, self.x)
        self.render()
    
    def plot_fft(self):
        self.ax.cla()
        lenght = len(self.t)
        yf = scipy.fft.fft(self.x - np.mean(self.x))
        xf = scipy.fft.fftfreq(lenght, 1 / self.fs)
        self.ax.plot(xf[:int(lenght/2)], np.abs(yf[:int(lenght/2)]))
        # self.ax.legend()
        self.render()
    
    def plot_stft(self):
        self.ax.cla()
        f, t, Zxx = scipy.signal.stft(self.x, self.fs, nperseg=200)
        f, Zxx = f[:int(len(Zxx)/2)], np.abs(Zxx[:int(len(Zxx)/2)])
        self.ax.pcolormesh(t*1000, f/1000, Zxx, vmin = 0, vmax = 0.1, shading = 'gouraud')
        # self.ax.legend()
        # self.ax.text('asdf',1, 1)
        # self.ax.set_title('STFT Magnitude')
        # self.ax.set_ylabel('Frequency [kHz]')
        # self.ax.set_xlabel('Time [ms]')
        self.render()
    
    def change_parameters(self, signal, fs):
        self.x = signal
        self.fs = fs
        self.t = np.arange(len(self.x))/self.fs


        
