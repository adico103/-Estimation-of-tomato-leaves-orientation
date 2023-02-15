'''
	Class to use the Turntable
	using serial port.
	Options:
		1. rot='e' - 'e' stands for 'eight', complete's 360 deg rotation in 8 iterations.
		2. rot='s' - 's' stands for 'sixsteen', complete's 360 deg rotation in 16 iterations. 
'''
import serial
import time

class Turntable():
	def __init__(self):
		self.ser = serial.Serial('COM4', baudrate=9600, timeout=1)	# change 'COM8' in linux!
		self.num_rots = 8
		self.i_rot = 0
		time.sleep(1)

	def rotate(self, rot='a', t_sleep=3):
		if rot == 'e':
			self.ser.write(b'e')
			self.num_rots = 8
			self.i_rot += 1
		else:
			self.ser.write(b'a')
			self.num_rots = 18
			self.i_rot += 1
		time.sleep(t_sleep)
		return False

	def closeSerial(self):
		self.ser.close()
