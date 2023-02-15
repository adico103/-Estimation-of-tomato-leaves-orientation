import matplotlib
matplotlib.use("Agg")
import cv2 as cv
import pyrealsense2 as rs
import pylab
import numpy as np

# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# rawdata = pjoin(dirname(scipy.io.__file__), currentdir, 'rawdata')

class Camera:
    def __init__(self,fake_camera):
        # self.fig = pylab.figure(figsize=[P_X/DPI, P_Y/DPI], # Inches
        #            dpi=DPI,        # 100 dots per inch, so the resulting buffer is 400x400 pixels
        #            )
        if not fake_camera:
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
        keep_trying=True
        while keep_trying:
            try:
                fs = self.pipeline.wait_for_frames()
                keep_trying=False
            except:
                keep_trying=True

        self.color_frame = fs.get_color_frame()
        self.depth_frame = fs.get_depth_frame()

        self.color_image = np.asanyarray(self.color_frame.get_data())
        self.depth_image = np.asanyarray(self.depth_frame.get_data())



    def save_img(self, name):
        path = "images//"
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
        cv.imwrite(path + name, self.color_image)
        # cv.imwrite(depth_name, self.depth_image)
        
        # save_color_image = cv.cvtColor(self.color_image, cv.COLOR_RGB2BGR)
        # cv.imwrite(os.path.join(path , depth_name), self.depth_image)




