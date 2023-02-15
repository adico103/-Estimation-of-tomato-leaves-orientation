# pyright: reportMissingImports=false, reportUndefinedVariable=false
''' Import dirs and inserts them to os ''' 
from gui.sprites import Camera
import dirs
import os
import sys
currentdir = dirs.current()
daq_dir = dirs.join(currentdir, 'daq', insert_path=True)
gui_dir = dirs.join(currentdir, 'gui', insert_path=True)
gui_img_dir = dirs.join(gui_dir, 'img', insert_path=True)
parent = os.path.dirname(currentdir)
sys.path.append(parent)
import params_update


''' Import my documents '''
from turntable import Turntable
from settings import *
from sprites import *
from ur5_kin import UR5_KIN
from ur5 import UR5_COM
from scipy.spatial.transform import Rotation


''' import relevant libraries '''
import socket
import time
import pygame as pg
import numpy as np
from pygame.locals import *
import feather
from numpy import pi
from numpy import cos, sin, tan, arctan2, arccos, arcsin, pi, sqrt

class Main:
	def __init__(self):

		pg.init()
		pg.mixer.init()
		self.screen = pg.display.set_mode((WIDTH, HEIGHT))
		self.surface = pg.display.get_surface()
		pg.display.set_caption(TITLE)
		self.bg = pg.image.load('gui/img/bg.png')
		# self.mask_ontop_camera = pg.image.load('gui/img/mask_ontop_camera.png')
		self.all_sprites = pg.sprite.Group()
		self.ur5_sprites = labelsGroup()
		self.clock = pg.time.Clock()
		self.running = True
		self.collect = False

		self.leaf_length = 0
		self.leaf_width = 0
		self.leaf_orientation_matrix = np.array([[0,0,0],[0,0,0],[0,0,0]])
		self.leaf_origin = np.array([0,0,0])
		self.leaf_orientation_vector = np.array([0,0,0])
		self.L3 = 96 # length of arm holding  the leaf [mm]
		self.L4 = 40 # distance between center of pole to leaf origin [mm] 
	
	def new(self):

		''' UR5 '''
		# Buttons
		self.b_ur5 = Button('button_ur5.png', (50, HEIGHT - 80), self.push_button)
		self.all_sprites.add(self.b_ur5)
		self.b_ur5_set = Button('button_ur5_set.png', (60, 70), self.push_button)
		self.all_sprites.add(self.b_ur5_set)
		self.b_ur5_test = Button('button_ur5_test.png', (60 + 80, 70), self.push_button)
		self.all_sprites.add(self.b_ur5_test)
		self.ur5_kin = UR5_KIN()
		self.pose_num = 0
		
		# Labels
		x_labels_ur5, y_labels_ur5, dist = 125, 110, 35
		xsize_labels_ur5 = 100
		self.l_radius = Label('Radius:', pos=(x_labels_ur5, y_labels_ur5, xsize_labels_ur5, 30), dim='[m]', init_text='[0.3,0.7,0.5]')
		self.ur5_sprites.add(self.l_radius)
		self.l_base = Label('Base:', pos=(x_labels_ur5, y_labels_ur5 + dist, xsize_labels_ur5, 30), dim='[m]', init_text='[1.2, 0.0]')
		self.ur5_sprites.add(self.l_base)
		self.l_steps = Label('Steps:', pos=(x_labels_ur5, y_labels_ur5 + dist*2, xsize_labels_ur5, 30), dim='[#]', init_text='6')
		self.ur5_sprites.add(self.l_steps)
		self.l_range = Label('Range:', pos=(x_labels_ur5, y_labels_ur5 + dist*3, xsize_labels_ur5, 30), dim='[deg]', init_text='[-30, 30]')
		self.ur5_sprites.add(self.l_range)
		self.push_button(button_name='ur5_set')
		

		''' Turntable '''
		# Buttons
		self.b_turntable = Button('button_turntable.png', (170, HEIGHT - 80), self.push_button)
		self.all_sprites.add(self.b_turntable)

		''' Camera '''
		# Buttons
		self.b_camera = Button('button_camera.png', (290, HEIGHT - 80), self.push_button)
		self.all_sprites.add(self.b_camera)
		self.camera_on = False

		self.run()


	def run(self):
		# Main loop
		self.playing = True
		while self.playing:
			self.clock.tick(FPS)
			self.events()
			self.update()
			self.draw()
			if self.collect:
				self.camera = Camera()
				self.daq()
			

	def update(self):
		pass


	def events(self):
		# Main loop - events
		for event in pg.event.get():
			# Check for closing window
			if event.type == pg.QUIT or pg.key.get_pressed()[pg.K_ESCAPE]:
				if self.playing:
					self.playing = False
				self.running = False

			# Check fo mouse click
			if event.type == pg.MOUSEBUTTONUP:
				pos = pg.mouse.get_pos()
				
				self.b_ur5.on_click(pos, button_name="ur5", text='pressed')
				self.b_ur5_set.on_click(pos, button_name="ur5_set", text='pressed')
				self.b_ur5_test.on_click(pos, button_name="ur5_test", text='pressed')
				self.b_turntable.on_click(pos, button_name="turntable", text='pressed')
				self.b_camera.on_click(pos, button_name="camera", text='pressed')

				self.ur5_sprites.on_click(pos)
			
			if event.type == pg.KEYDOWN:
				self.ur5_sprites.change_text(event)

				keystate = pg.key.get_pressed() # Get which key is pressed

				if keystate[pg.K_s]:
					self.collect = True
			

	def draw(self):
		# Main loop draw
		self.screen.fill(BG)

		if self.camera_on:
			self.camera.play_camera()
			surf = pg.image.fromstring(self.camera.raw_data, self.camera.canvas.get_width_height(), 'RGB')
			self.screen.blit(surf, (390, -25))
			
		self.screen.blit(self.bg, (0, 0))
		self.all_sprites.draw(self.screen)
		self.ur5_sprites.draw(self.screen)
		# *after* drawing everything, flip the display
		pg.display.flip()
	
	def get_leaf_pose( self, p , primary_orientation , r ):
		th_rot = -45*np.pi/180
		L_1 = self.d_base[0]
		L_2 = self.d_base[1]
		L_3 = self.L3/1000
		L_4 = self.L4/1000
		alpha = 0.5*pi-(float(primary_orientation) * pi* 0.25 )
		theta = pi*r/9.0
		xR, yR, zR, rxR, ryR, rzR = p
		P_rotation = Rotation.from_rotvec(np.array([rxR, ryR, rzR]))

		T_0ֹֹֹ00 = np.array([[1, 0, 0, 0],
							[0, 0, 1, 0],
							[0, -1, 0, 0],
							[0, 0, 0, 1]])

		T_A1 = np.array([[cos(th_rot), -sin(th_rot), 0, 0],
							[sin(th_rot), cos(th_rot), 0, 0],
							[0, 0, 1, 0],
							[0, 0, 0, 1]])

		T_temp = np.array(P_rotation.as_matrix())

		vec_loc= np.array([[xR, yR, zR]]).T
		T_A00 = np.concatenate((T_temp, vec_loc), axis=1)
		T_A00 = np.concatenate((T_A00, np.array([[0,0,0,1]])), axis=0)
		T_00A = np.linalg.inv(T_A00)
		T_001 = np.dot(T_00A,T_A1)
		T_01 = np.dot(T_0ֹֹֹ00,T_001)
		T_12 = np.array([[cos(theta), -sin(theta), 0, 0],
					[sin(theta), cos(theta), 0, L_1],
					[0, 0, 1, 0],
					[0, 0, 0, 1]])
		T_23 = np.array([[1, 0, 0, 0],
					[0, cos(alpha), -sin(alpha), 0],
					[0, sin(alpha), cos(alpha), L_2],
					[0, 0, 0, 1]])
		T_3Leaf =np.array([[0, -1, 0, -L_4],
					[0, 0, 1, L_3],
					[-1, 0, 0, 0],
					[0, 0, 0, 1]])
		T_0Leaf = np.dot(T_01, T_12)
		T_0Leaf = np.dot(T_0Leaf, T_23)
		T_0Leaf = np.dot(T_0Leaf, T_3Leaf)

		leaf_origin = np.dot(T_0Leaf, np.array([0,0,0,1]).T)[:3]
		orientation_matrix = np.array(T_0Leaf[:3,:3])
		rv_temp = Rotation.from_matrix(orientation_matrix)
		rotation_vector = rv_temp.as_rotvec()
		return orientation_matrix ,leaf_origin, rotation_vector


	def daq(self):
		# self.camera = Camera()

		error_flag = True
		while error_flag:
			new_leaf = input("New Leaf? (y/n)")
			if (new_leaf == 'y'):
				self.leaf_number = input("Enter leaf_number :")
				self.leaf_length = input("Enter leaf_length (mm) :")
				self.leaf_width = input("Enter leaf_width (mm) :")

			if (new_leaf == 'n' or new_leaf == 'y') :
				while error_flag:
					primary_orientation = input("Enter primary_orientation (0: 0 degrees , 1: 45 degrees, 2:90 degrees): ")
					if (primary_orientation!= '0' and primary_orientation!= '1' and primary_orientation!= '2'):
						print('Try Again')
					else :
						error_flag = False
			elif (new_leaf != 'n' and new_leaf != 'y') : 
				if error_flag :
					print('Try Again')		
		 

		for radius in self.r_plant:
			
	
			self.ur5_kin.define_geometry(radius, d_base=self.d_base, th_tcp_steps=self.th_tcp_steps, th_tcp_range=self.th_tcp_range)
			self.p = self.ur5_kin.get_rotational_path()
			if radius == 0.7:
				move = ("movej(["+("%f,%f,%f,%f,%f,%f"%(0, pi*0.3, 0, 0, 0, 0)) +"], a=0.05, v=0.2, r=0)" +"\n").encode("utf8")
				self.s.send(move)
				time.sleep(10)
				pass
			
			for i, p in enumerate(self.p):

				self.ur5.move(p)
				time.sleep(3)
				if (i!=1 and radius==0.3):
					time.sleep(3)

				
				for r in range(12):

					self.leaf_orientation_matrix , self.leaf_origin, self.leaf_orientation_vector = self.get_leaf_pose(p,primary_orientation ,r )
					name = 'N{}_d{}_p{}_r{}_O{}.png'.format(self.leaf_number,radius, np.around(self.ur5_kin.th_tcp_vec[i]*180/pi,1), r, primary_orientation)	
					# print(name)
					self.camera.play_camera()
					self.camera.save_img(name ,self.leaf_length,self.leaf_width,self.leaf_orientation_matrix, self.leaf_origin ,self.leaf_orientation_vector, p,params_update.exp_folder )
					print('dis: {}, angle:{}, rot:{}/360'.format( radius,np.around(self.ur5_kin.th_tcp_vec[i]*180/pi,0),r*360/12))
					self.turntable.rotate('e', t_sleep=2)
					self.draw()
					
				
		# self.collect = False
	
	def push_button(self, button_name, text=None, button=None):
		
		''' UR5 '''
		if button_name == 'ur5':
			if button_name == 'ur5':
				HOST = "192.168.1.113"  # The remote host
				PORT_30003 = 30003
				print("Starting Program")
				try:
					self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
					self.s.settimeout(5)
					self.s.connect((HOST, PORT_30003))
					time.sleep(0.1)
					self.ur5 = UR5_COM(self.s)
					self.b_ur5.image.fill((255, 255, 255, 128), None, pg.BLEND_RGBA_MULT)
					self.b_ur5.clicked_enabled = False
				except socket.error as socketerror:
					print("Error: ", socketerror)
		
		if button_name == 'ur5_set':
			
			if self.ur5_sprites.check_text() == False:
				print('Check the boxes!')
				return
			
			self.r_plant, self.d_base, self.th_tcp_steps, self.th_tcp_range = self.get_ur5_parameters()
			# print('r_plant',self.r_plant )
			self.ur5_kin.define_geometry(self.r_plant[0], d_base=self.d_base, th_tcp_steps=self.th_tcp_steps, th_tcp_range=self.th_tcp_range)
			self.p = self.ur5_kin.get_rotational_path()
		
		if button_name == 'ur5_test':

			self.ur5.move(self.p[self.pose_num])
			self.pose_num = (self.pose_num + 1) % self.ur5_kin.th_tcp_steps
			print(self.pose_num)
			
			# print(self.p)

		
		''' Turntable '''
		if button_name == 'turntable':
			try:
				self.turntable = Turntable()
				self.b_turntable.image.fill((255, 255, 255, 128), None, pg.BLEND_RGBA_MULT)
				self.b_turntable.clicked_enabled = False
				self.b_turntable.clicked = True
			# except (AttributeError, serial.serialutil.SerialException):
			except (AttributeError):
				print('Turntable disconnected!')
		
		''' Camera '''
		if button_name == 'camera':
			try:
				self.camera = Camera()
				self.camera_on = True
			except RuntimeError:
				print('Camera disconnected')



			
		print(button_name)


	def get_ur5_parameters(self):

		r_plant = self.l_radius.user_text
		print('r_plant',r_plant )
		if self.l_radius.user_text.startswith('['):
			r_plant = np.fromstring(r_plant[1:-1], dtype=float, sep=',')
		else:
			r_plant = np.fromstring(r_plant, dtype=float, sep=',')


		d_base = self.l_base.user_text
		if d_base[0] == '[':
			d_base = np.fromstring(d_base[1:-1], dtype=float, sep=',')
		else:
			d_base = np.fromstring(d_base, dtype=float, sep=',')

		th_tcp_steps = int(self.l_steps.user_text)

		th_tcp_range = self.l_range.user_text
		if th_tcp_range[0] == '[':
			th_tcp_range = np.fromstring(th_tcp_range[1:-1], dtype=float, sep=',')
		else:
			th_tcp_range = np.fromstring(th_tcp_range, dtype=float, sep=',')

		return r_plant, d_base, th_tcp_steps, th_tcp_range
        


main = Main()

while main.running:
    main.new()

pg.quit()
