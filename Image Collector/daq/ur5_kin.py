'''
Kinemtaics of UR5 arm for Sonar orientation
Based on 'Ching-Yen Weng'
'''

import numpy as np
from numpy import cos, sin, tan, arctan2, arccos, arcsin, pi, sqrt
from numpy import linalg as LA
from numpy.linalg import norm
from math import radians, degrees
from scipy.spatial.transform import Rotation
import cmath

class UR5_KIN():
    def __init__(self):
        # Distances (unit: mm)
        self.d1 = 0.0892
        self.d2 = self.d3 = 0
        self.d4 = 0.1093
        self.d5 = 0.09475
        self.d6 = 0.0825

        # a (unit: mm)
        self.a1 = self.a4 = self.a5 = self.a6 = 0
        self.a2 = -0.425
        self.a3 = -0.392

        # List type of D-H parameter
        self.alpha = np.array([90, 0, 0, 90, -90, 0])*np.pi/180
        self.a = np.array([self.a1, self.a2, self.a3, self.a4, self.a5, self.a6])
        self.d = np.array([self.d1, self.d2, self.d3, self.d4, self.d5, self.d6])

        # Variables for position and angles of Arm accordance to Plant
        self.d_base = None
        self.th_tcp_steps = None
        self.th_tcp_range = None
        self.th_tcp_vec = None
        self.r_plant = None
        self.p = np.empty([0, 6])
        self.th_path = np.empty([0, 6])
        self.d_z = None
        self.d_x = None
        self.num_of_poses = None

        # Variables for position according to sonar
        self.tcp_sonar = 0.2 # distance from end of speaker to tcp

    def transformMatrix(self, i, th):
        Rot_z = np.matrix(np.identity(4))
        Rot_z[0, 0] = Rot_z[1, 1] = cos(th[i])
        Rot_z[0, 1] = -sin(th[i])
        Rot_z[1, 0] = sin(th[i])

        Trans_z = np.matrix(np.identity(4))
        Trans_z[2, 3] = self.d[i]

        Trans_x = np.matrix(np.identity(4))
        Trans_x[0, 3] = self.a[i]

        Rot_x = np.matrix(np.identity(4))
        Rot_x[1, 1] = Rot_x[2, 2] = cos(self.alpha[i])
        Rot_x[1, 2] = -sin(self.alpha[i])
        Rot_x[2, 1] = sin(self.alpha[i])

        A_i = Rot_z @ Trans_z @ Trans_x @ Rot_x
            
        return A_i

    def forwardKinematics(self, theta, i_unit='r', o_unit='n'):
        T_06 = np.matrix(np.identity(4))

        if i_unit == 'd':
            theta = [radians(i) for i in theta]
        
        for i in range(6):
            T_06 = T_06 @ self.transformMatrix(i, theta)

        if o_unit == 'n':
            return T_06
           
    def inverseKinematics(self, T_06, i_unit='r', o_unit='r'):
        # Initialization of a set of feasible solutions
        theta = np.zeros((8, 6))
    
        # theta1
        P_05 = T_06[0:3, 3] - self.d6 * T_06[0:3, 2]
        phi1 = arctan2(P_05[1], P_05[0])
        if 1 - abs(self.d4 / sqrt(P_05[0] ** 2 + P_05[1] ** 2)) < 1e-5:
            phi2 = 0
        else:
            phi2 = arccos(self.d4 / sqrt(P_05[0] ** 2 + P_05[1] ** 2))
        theta1 = [pi / 2 + phi1 + phi2, pi / 2 + phi1 - phi2]
        theta[0:4, 0] = theta1[0]
        theta[4:8, 0] = theta1[1]
    
        # theta5
        P_06 = T_06[0:3, 3]
        theta5 = []
        for i in range(2):
            theta5.append(arccos((P_06[0] * sin(theta1[i]) - P_06[1] * cos(theta1[i]) - self.d4) / self.d6))
        for i in range(2):
            theta[2*i, 4] = theta5[0]
            theta[2*i+1, 4] = -theta5[0]
            theta[2*i+4, 4] = theta5[1]
            theta[2*i+5, 4] = -theta5[1]
    
        # theta6
        T_60 = np.linalg.inv(T_06)
        theta6 = []
        for i in range(2):
            for j in range(2):
                s1 = sin(theta1[i])
                c1 = cos(theta1[i])
                s5 = sin(theta5[j])
                theta6.append(arctan2((-T_60[1, 0] * s1 + T_60[1, 1] * c1) / s5, (T_60[0, 0] * s1 - T_60[0, 1] * c1) / s5))
        for i in range(2):
            theta[i, 5] = theta6[0]
            theta[i+2, 5] = theta6[1]
            theta[i+4, 5] = theta6[2]
            theta[i+6, 5] = theta6[3]

        # theta3, theta2, theta4
        for i in range(8):  
            # theta3
            T_46 = self.transformMatrix(4, theta[i]) @ self.transformMatrix(5, theta[i])
            T_14 = np.linalg.inv(self.transformMatrix(0, theta[i])) @ T_06 @ np.linalg.inv(T_46)
            P_13 = T_14 @ np.array([[0, -self.d4, 0, 1]]).T - np.array([[0, 0, 0, 1]]).T
            if i in [0, 2, 4, 6]:
                theta[i, 2] = -cmath.acos((np.linalg.norm(P_13) ** 2 - self.a2 ** 2 - self.a3 ** 2) / (2 * self.a2 * self.a3)).real
                theta[i+1, 2] = -theta[i, 2]
                
            # theta2
            theta[i, 1] = -arctan2(P_13[1], -P_13[0]) + arcsin(self.a3 * sin(theta[i, 2]) / np.linalg.norm(P_13))

            # theta4
            T_13 = self.transformMatrix(1, theta[i]) @ self.transformMatrix(2, theta[i])
            T_34 = np.linalg.inv(T_13) @ T_14
            theta[i, 3] = arctan2(T_34[1, 0], T_34[0, 0])       

        # Select the prefereble variation of joints angle
        theta = theta[4, :]

        q_sol = theta
        # Output format
        if o_unit == 'r': # (unit: radian)
            return q_sol
        elif o_unit == 'd': # (unit: degree)
            return [degrees(i) for i in q_sol]

    def rot2euler(self, R):
        beta = arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
        alpha = arctan2(R[1, 0]/cos(beta), R[0, 0]/cos(beta))
        gama = arctan2(R[2, 1]/cos(beta), R[2, 2]/cos(beta))
        return np.array([alpha, beta, gama])
    
    def rot2vec(self, R):
        theta = arccos((np.trace(R) - 1)/(2))
        u = (1/(2*sin(theta)))*np.array([R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]])
        return u*theta
    
    def get_tcp_wanted_rot(self, th_rot, d, th_tcp, r_plant):
        # initiate plant position
        p = np.zeros([4, 4])
        T_01 = np.array([[cos(th_rot), -sin(th_rot), 0, 0],
                        [sin(th_rot), cos(th_rot), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        T_12 = np.array([[0, 0, 1, 0],
                        [0, 1, 0, 0],
                        [-1, 0, 0, 0],
                        [0, 0, 0, 1]])
        T_23 = np.array([[1, 0, 0, -d[1]],
                        [0, 1, 0, 0],
                        [0, 0, 1, d[0]],
                        [0, 0, 0, 1]])
        # T_34 = np.array([[1, 0, 0, 0],
        #                 [0, cos(th_tcp), sin(th_tcp), 0],
        #                 [0, -sin(th_tcp), cos(th_tcp), 0],
        #                 [0, 0, 0, 1]])
        T_34 = np.array([[cos(th_tcp), 0, sin(th_tcp), 0],
                         [0, 1, 0, 0],
                         [-sin(th_tcp), 0, cos(th_tcp), 0],
                         [0, 0, 0, 1]])
        T_45 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, -r_plant],
                        [0, 0, 0, 1]])
        T_56 = np.array([[cos(pi/2), sin(pi/2), 0, 0],
                        [sin(-pi/2), cos(pi/2), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        # T_56 = np.array([[cos(pi/4), sin(pi/4), 0, 0],
        #         [sin(-pi/4), cos(-pi/4), 0, 0],
        #         [0, 0, 1, 0],
        #         [0, 0, 0, 1]])                
        T = np.dot(T_01, T_12)
        T = np.dot(T, T_23)
        T = np.dot(T, T_34)
        T = np.dot(T, T_45)
        T = np.dot(T, T_56)
        return T

    def get_rotational_path(self, th_rot=45*np.pi/180, move_type='pose'):
        if move_type == 'pose':
            self.p = np.empty([0, 6])
            for th_tcp in self.th_tcp_vec:
                T_temp = self.get_tcp_wanted_rot(th_rot, self.d_base, th_tcp, self.r_plant)
                p_temp = T_temp[:3, 3]
                rv_temp = Rotation.from_matrix(T_temp[:3, :3])
                rv_temp = rv_temp.as_rotvec()
                self.p = np.vstack([self.p, np.hstack([p_temp[:], rv_temp[:]])])
            return self.p

        if move_type == 'joints':
            self.th_path = np.empty([0, 6])
            for th_tcp in self.th_tcp_vec:
                T_temp = self.get_tcp_wanted_rot(th_rot, self.d_base, th_tcp, self.r_plant)
                th_temp = self.inverseKinematics(T_temp)
                self.th_path = np.vstack([self.th_path, th_temp])
            return self.th_path

    def define_geometry(self, r_plant, d_base=None, th_tcp_steps=None, th_tcp_range=None, d_x=None, d_z=None):
        self.r_plant = r_plant
        if isinstance(th_tcp_steps, type(None)) is False:
            self.d_base = d_base
            self.th_tcp_steps = th_tcp_steps
            self.th_tcp_range = th_tcp_range
            if r_plant == 0.7:
                self.th_tcp_range[0] = 30
                self.th_tcp_range[1]= -30
                self.th_tcp_steps = 5
            if r_plant == 0.5:
                self.th_tcp_range[0] = 20
                self.th_tcp_range[1]= -20
                self.th_tcp_steps = 3
            if r_plant < 0.4:
                self.th_tcp_range[0] = -10
                self.th_tcp_range[1]= 10
                self.th_tcp_steps = 3
                


            self.th_tcp_vec = np.linspace(self.th_tcp_range[0], self.th_tcp_range[1], self.th_tcp_steps)*pi/180

            self.num_of_poses = th_tcp_steps
            return
        self.d_z = d_z
        self.d_x = d_x
        self.num_of_poses = len(self.d_z)
        
    
    def get_tcp_wanted_z(self, th_rot, d_z, r_plant):
        # Initiate plant position
        p = np.zeros([4, 4])
        # Rotating base by th_rot, default - 45 [deg]
        T_01 = np.array([[cos(th_rot), -sin(th_rot), 0, 0],
                         [sin(th_rot), cos(th_rot), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        # Alternation - X axis -> -Z, Z axis -> +X 
        T_12 = np.array([[0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [-1, 0, 0, 0],
                         [0, 0, 0, 1]])
        # Move tcp - heigth   : tomato height from base - (d_z).
        #          - distance : end of speaker from tomato - (d_x-r_plant-self.tcp_sonar).
        T_23 = np.array([[1, 0, 0, -d_z],
                    [0, 1, 0, 0],
                    [0, 0, 1, self.d_x-r_plant],
                    [0, 0, 0, 1]])
        # T_23 = np.array([[1, 0, 0, -d_z],
        #                  [0, 1, 0, 0],
        #                  [0, 0, 1, self.d_x-r_plant-self.tcp_sonar],
        #                  [0, 0, 0, 1]])
        T_34 = np.array([[cos(pi/4), -sin(pi/4), 0, 0],
                         [sin(pi/4), cos(pi/4), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        T = np.dot(T_01, T_12)
        T = np.dot(T, T_23)
        T = np.dot(T, T_34)
        return T
    
    def get_disp_path(self, th_rot=45*pi/180, move_type='pose'):
        if move_type == 'pose':
            self.p = np.empty([0, 6])
            for d_z in self.d_z:
                T_temp = self.get_tcp_wanted_z(th_rot, d_z, self.r_plant)
                p_temp = T_temp[:3, 3]
                rv_temp = Rotation.from_matrix(T_temp[:3, :3])
                rv_temp = rv_temp.as_rotvec()
                self.p = np.vstack([self.p, np.hstack([p_temp[:], rv_temp[:]])])
            return self.p
