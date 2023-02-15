import sys
sys.path.append("/home/ur5/code/") # change according to path
import socket
import time
import numpy as np
import struct
from math import pi
from codecs import decode, encode

class UR5_COM():

    def __init__(self, sock=None):
        self.home_status = 0
        self.program_run = 0
        self.x, self.y, self.z, self.Rx, self.Ry, self.Rz = float, float, float, float, float, float
        self.xR, self.yR, self.zR, self.rxR, self.ryR, self.rzR = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        if sock == None:
            HOST = "192.168.1.113" # The remote host
            PORT_30003 = 30003

            print ("Starting Program")
            try:
                self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.s.settimeout(10)
                self.s.connect((HOST, PORT_30003))
                time.sleep(0.1)
                self.write_init()
                self.read_init()
            except socket.error as socketerror:
                print("Error: ", socketerror)
        else:
            self.s = sock
            time.sleep(0.1)
            self.write_init()
            self.read_init()

    '''
        Reading functions
    '''
    def read_init(self):
        #3-8 Returns the desired angular position of all joints
        self.packet_1 = self.s.recv(4)
        self.packet_2 = self.s.recv(8)
        self.packet_3 = self.s.recv(8)
        self.packet_4 = self.s.recv(8)
        self.packet_5 = self.s.recv(8)
        self.packet_6 = self.s.recv(8)
        self.packet_7 = self.s.recv(8)
        self.packet_8 = self.s.recv(8)
        #9-14 Returns the desired angular velocities of all joints
        self.packet_9 = self.s.recv(8)
        self.packet_10 = self.s.recv(8)
        self.packet_11 = self.s.recv(8)
        self.packet_12 = self.s.recv(8)
        self.packet_13 = self.s.recv(8)
        self.packet_14 = self.s.recv(8)
        # 15-20 Returns the desired angular accelerations of all joints
        self.packet_15 = self.s.recv(8)
        self.packet_16 = self.s.recv(8)
        self.packet_17 = self.s.recv(8)
        self.packet_18 = self.s.recv(8)
        self.packet_19 = self.s.recv(8)
        self.packet_20 = self.s.recv(8)
        # 21-26 Returns the desired Currents of all joints -- not used
        self.packet_21 = self.s.recv(8)
        self.packet_22 = self.s.recv(8)
        self.packet_23 = self.s.recv(8)
        self.packet_24 = self.s.recv(8)
        self.packet_25 = self.s.recv(8)
        self.packet_26 = self.s.recv(8)
        # 27-32 Returns the desired Torques of all joints
        self.packet_27 = self.s.recv(8)
        self.packet_28 = self.s.recv(8)
        self.packet_29 = self.s.recv(8)
        self.packet_30 = self.s.recv(8)
        self.packet_31 = self.s.recv(8)
        self.packet_32 = self.s.recv(8)
        # 33 Returns the same information as packages 3-8 -- not used again
        self.packet_33 = self.s.recv(48)
        # 34 Returns the same information as packages 9-14 -- not used again
        self.packet_34 = self.s.recv(48)
        # 35 Returns the same information as packages 15-20 -- not used again
        self.packet_35 = self.s.recv(48)
        # 36 Returns the same information as packages 21-26 -- not used
        self.packet_36 = self.s.recv(48)
        # 37-42 Returns actual cartesian coordinates of the tool
        self.packet_37 = self.s.recv(8)
        self.packet_38 = self.s.recv(8)
        self.packet_39 = self.s.recv(8)
        self.packet_40 = self.s.recv(8)
        self.packet_41 = self.s.recv(8)
        self.packet_42 = self.s.recv(8)
        # 43- 48 Returns actual speed of the tool given in cartesian coordinates.The first three values are the
        #cartesian speeds along x, y, z. and the last three define the current rotation axis, rx, ry, rz, and the
        #length | rz, ry, rz | defines the angular velocity in radians / s.
        self.packet_43 = self.s.recv(8)
        self.packet_44 = self.s.recv(8)
        self.packet_45 = self.s.recv(8)
        self.packet_46 = self.s.recv(8)
        self.packet_47 = self.s.recv(8)
        self.packet_48 = self.s.recv(8)
        # 49 Returns the generalized force on the tcp -- not used
        self.packet_49 = self.s.recv(48)
        # 50 Returns Target Cartesian coordinates of the tool: (x,y,z,rx,ry,rz), where rx, ry and rz is a rotation
        # vector representation of the tool orientation --not used
        self.packet_50 = self.s.recv(48)
        #51 Returns Target speed of the tool given in Cartesian coordinates -- not used
        self.packet_51 = self.s.recv(48)
        # 52 Returns Current state of the digital inputs. NOTE: these are bits encoded as int64_t,
        # e.g. a value of 5 corresponds to bit 0 and bit 2 set high
        self.packet_52 = self.s.recv(8)
        # 53 Returns Temperature of each joint in degrees celsius -- not used
        self.packet_53 = self.s.recv(48)
        # 54 Returns Controller realtime thread execution time
        self.packet_54 = self.s.recv(8 )
        # 55 Returns Test Value - A value used by Universal Robots software only -- not used
        self.packet_55 = self.s.recv(8)
        # 56 Returns robot mode - see DataStreamFromURController in the Excel file for more information -- not used
        self.packet_56 = self.s.recv(8)
        # 57 Returns Joint control mode -see DataStreamFromURController in the Excel file for more information -- not
        #used
        self.packet_57= self.s.recv(48)
        # 58 Returns Safety mode - see	DataStreamFromURController in the Excel file for more information -- not used
        self.packet_58 = self.s.recv(8)
        # 59 is Used by Universal Robots software only
        self.packet_59 = self.s.recv(48)
        # 60-62 Returns the current reading of the tool accelerometer as a three-dimensional vector.
        # The accelerometer axes are aligned with the tool coordinates, and pointing an axis upwards
        # results in a positive reading.
        self.packet_60 = self.s.recv(8)
        self.packet_61 = self.s.recv(8)
        self.packet_62 = self.s.recv(8)
        # 63 is Used by Universal Robots software only
        self.packet_63 = self.s.recv(48)
        # 64 Returns Speed scaling of the trajectory limiter -- not used
        self.packet_64 = self.s.recv(8)
        # 65 Returns Norm of Cartesian linear momentum --not used
        self.packet_65 = self.s.recv(8)
        # 66&67 are Used by Universal Robots software only
        self.packet_66 = self.s.recv(8)
        self.packet_67 = self.s.recv(8)
        # 68 Returns the main voltage -- not used
        self.packet_68 = self.s.recv(8)
        # 69 Returns the robot voltage (48V) -- not used
        self.packet_69 = self.s.recv(8)
        # 70 Returns the robot current --not used
        self.package_70 = self.s.recv(8)
        # 71 Returns Actual joint voltages -- not used
        self.package_71 = self.s.recv(48)
        # 72 Returns digital outputs --not used
        self.package_72 = self.s.recv(8)
        # 73 Returns program state --not used
        self.package_73 = self.s.recv(8)
    
    def get_act_joint_pos(self):
        base = struct.unpack('!d',self.packet_3)[0] % (2*pi)
        # print ("Base in Deg = ",base*(180/pi) )
        shoulder = ((struct.unpack('!d',self.packet_4)[0] - 2*pi) % (2*pi))
        # print ("Shoulder in Deg = ",shoulder*(180/pi))
        elbow = struct.unpack('!d',self.packet_5)[0] % (2*pi) - 2*pi
        # print ("Elbow in Deg = ", elbow * (180 / pi))
        wrist1 = ((struct.unpack('!d',self.packet_6)[0]) % (2*pi))
        # print ("Wrist1 in Deg = ", wrist1 * (180 / pi))
        wrist2 = struct.unpack('!d',self.packet_7)[0] % (2*pi)
        # print ("Wrist2 in Deg = ", wrist2 * (180 / pi))
        wrist3 = struct.unpack('!d',self.packet_8)[0] % (2*pi)
        # print ("Wrist3 in Deg = ", (wrist3 * (180 / pi)))
        joint_angles = np.array([base, shoulder, elbow, wrist1, wrist2, wrist3])
        return joint_angles

    def get_act_joint_vel(self):
        baseV = struct.unpack('!d',self.packet_9)[0]
        print ("Base Velocity in rad\sec = ", baseV )
        shoulderV = struct.unpack('!d',self.packet_10)[0]
        print ("Shoulder Velocity in rad\sec = ",shoulderV)
        elbowV = struct.unpack('!d',self.packet_11)[0]
        print ("Elbow Velocity in rad\sec = ", elbowV)
        wrist1V = struct.unpack('!d',self.packet_12)[0]
        print ("Wrist1 Velocity in rad\sec = ", wrist1V)
        wrist2V = struct.unpack('!d',self.packet_13)[0]
        print ("Wrist2 Velocity in rad\sec = ", wrist2V)
        wrist3V = struct.unpack('!d',self.packet_14)[0]
        print ("Wrist3 Velocity in rad\sec= ", wrist3V)

    def get_act_joint_a(self):
        baseA = struct.unpack('!d', self.packet_15)[0]
        print ("Base Accelerations =", baseA)
        shoulderA = struct.unpack('!d', self.packet_16)[0]
        print ("Shoulder Acceleration = ", shoulderA)
        elbowA = struct.unpack('!d', self.packet_17)[0]
        print ("Elbow Acceleration = ", elbowA)
        wrist1A = struct.unpack('!d', self.packet_18)[0]
        print ("Wrist1 Acceleration = ", wrist1A)
        wrist2A = struct.unpack('!d', self.packet_19)[0]
        print ("Wrist2 Acceleration = ", wrist2V)
        wrist3A = struct.unpack('!d', self.packet_20)[0]
        print ("Wrist3 Acceleration= ", wrist3A)

    def get_act_torques(self):
        baseT = struct.unpack('!d',self.packet_27)[0]
        print ("Base Torque in NM =", baseT)
        shoulderT = struct.unpack('!d',self.packet_28)[0]
        print ("Shoulder Torque in NM = ", shoulderT)
        elbowT = struct.unpack('!d',self.packet_29)[0]
        print ("Elbow Torque in NM = ", elbowT)
        wrist1T = struct.unpack('!d',self.packet_30)[0]
        print ("Wrist1 Torque in NM = ", wrist1T)
        wrist2T = struct.unpack('!d',self.packet_31)[0]
        print ("Wrist2 Torque in NM = ", wrist2T)
        wrist3T = struct.unpack('!d',self.packet_32)[0]
        print ("Wrist3 Torque in NM= ", wrist3T)

    def get_tcp_position(self): # IN BASE COORDINATES :
        # coef_fix = -1
        # self.read_init() # read all the binary data
        x = struct.unpack('!d', self.packet_37)[0]
        # print ("X in mm = ", x * 1000)
        y = struct.unpack('!d', self.packet_38)[0]
        # print ("Y in mm = ", y * 1000)
        z = struct.unpack('!d', self.packet_39)[0]
        # print ("Z in mm = ", z * 1000)
        Rx = struct.unpack('!d', self.packet_40)[0]
        # print ("Rx in rad = ", Rx)
        Ry = struct.unpack('!d', self.packet_41)[0]
        # print ("Ry in rad = ", Ry)
        Rz = struct.unpack('!d', self.packet_42)[0]
        # print ("Rz in rad = ", Rz)
        # return(np.array([x,y,z,Rx*coef_fix,Ry*coef_fix,Rz*coef_fix]))
        return(np.array([x,y,z,Rx,Ry,Rz]))

    def get_tcp_velocities(self):
        xV = struct.unpack('!d', self.packet_43)[0]
        print ("X Velocity  = ", xV)
        yV = struct.unpack('!d', self.packet_44)[0]
        print ("Y Velocity = ", yV)
        zV = struct.unpack('!d', self.packet_45)[0]
        print ("Z Velocity = ", zV)
        RxV = struct.unpack('!d', self.packet_46)[0]
        print ("Rx Velocity = ", RxV)
        RyV = struct.unpack('!d', self.packet_47)[0]
        print ("Ry Velocity = ", RyV)
        RzV = struct.unpack('!d', self.packet_48)[0]
        print ("Rz Velocity = ", RzV)

    def Reading_tool_accelerometer(self):
        xacc = struct.unpack('!d',self.packet_60)[0]
        print ("X tool accelerometer in m\s^2  = ", xacc)
        yacc = struct.unpack('!d',self.packet_61)[0]
        print ("Y tool accelerometer in m\s^2  = ", yacc)
        zacc = struct.unpack('!d',self.packet_62)[0]
        print ("Z tool accelerometer in m\s^2 = ", zacc)

        self.home_status = 1
        self.program_run = 0
        self.s.close()

    '''
        Writing functions
    '''
    def write_init(self):
        self.s.send(("set_gravity([0.0, 0.0, 9.82])" + "\n").encode('utf8'))
        time.sleep(0.1)
        self.s.send(("set_tool_voltage(0)" + "\n").encode('utf8'))
        time.sleep(0.1)
        self.s.send(("set_safety_mode_transition_hardness(1)" + "\n").encode('utf8'))
        time.sleep(0.1)
        # self.s.send(("set_tcp(p[0.0,0.0,0.0,0.0,0.0,0.0])" + "\n").encode('utf8')) # maybe change 
        self.s.send(("set_tcp(p["+("%f,%f,%f,%f,%f,%f"%(self.xR, self.yR, self.zR, self.rxR, self.ryR, self.rzR)) +"])" + "\n").encode('utf8')) # maybe change ^
        time.sleep(0.1)
        self.s.send(("set_payload(1.0)" + "\n").encode('utf8'))
        time.sleep(0.1)
        self.s.send(("set_standard_analog_input_domain(0, 1)" + "\n").encode('utf8'))
        time.sleep(0.1)
        self.s.send(("set_standard_analog_input_domain(1, 1)" + "\n").encode('utf8'))
        time.sleep(0.1)
        self.s.send(("set_tool_analog_input_domain(0, 1)" + "\n").encode('utf8'))
        time.sleep(0.1)
        self.s.send(("set_tool_analog_input_domain(1, 1)" + "\n").encode('utf8'))
        time.sleep(0.1)
        self.s.send(("set_analog_outputdomain(0, 0)" + "\n").encode('utf8'))
        time.sleep(0.1)
        self.s.send(("set_analog_outputdomain(1, 0)" + "\n").encode('utf8'))
        time.sleep(0.1)
        self.s.send(("set_input_actions_to_default()" + "\n").encode('utf8'))
        time.sleep(0.1)
        self.s.settimeout(10)
        time.sleep(1.00)

    def print_pose(self):
        pose = self.get_tcp_position()
        print("X =", np.around(pose[0] * 1000, 3), "[mm]")
        print("Y =", np.around(pose[1] * 1000, 3), "[mm]")
        print("Z =", np.around(pose[2] * 1000, 3), "[mm]")
        print("Rx =", np.around(pose[3], 4), "[rad]")
        print("Ry =", np.around(pose[4], 4), "[rad]")
        print("Rz =", np.around(pose[5], 4), "[rad]")


    def move(self, pose, move_type='pose'):
        xR, yR, zR, rxR, ryR, rzR = pose
        if move_type == 'pose':
            move = ("movej(p["+("%f,%f,%f,%f,%f,%f"%(xR, yR, zR, rxR, ryR, rzR)) +"], a=0.4, v=1.2, r=0)" +"\n").encode("utf8")
        if move_type == 'joints':
            move = ("movej(["+("%f,%f,%f,%f,%f,%f"%(xR, yR, zR, rxR, ryR, rzR)) +"], a=0.2, v=0.4, r=0)" +"\n").encode("utf8")
        self.s.send(move)
        time.sleep(2)
        pass
