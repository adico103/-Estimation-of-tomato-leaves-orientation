import time
from UR5_robot import Robot
import numpy as np

realtime = False
current_measure = True

########################################
######### before RealTime #########
### 1. change to false ###
if realtime:
    fake_robot = False
    fake_camera = False
else:
    fake_robot = True
    fake_camera = True
best_route = [40,-35] # Best Elevation Angles
r_plant = 0.5
## 2. update start pose if you want to start somewhere else, when the leaf is set to be at distance r_plant [m] from end effector at this start pose###
start_pose_found = ([-0.4590162 , -0.39949618,  0.04471388,  0.63762857, -1.47512431, 0.63749924])

######### before RealTime #########
########################################

feather_folder = 'data\\'
move_to_center = False
change_z = False
detect = False
num_exp = 5
sub_exp = 5


def findRealtimeLeafOrientation(robot):
    
    robot.start(start_pose_found,move_to_center,change_z,detect = detect) # find leaf, find leaf center, adjust distance
    robot.poses = []
    robot.r_plant = r_plant
    robot.plan_route() # plan route for motion
    robot.take_img(index=0,fake_camera=fake_camera,exmple_file=feather_folder+'Pose_0.feather') # Take first img
    name = 'Pose_0'
    print('Saving pose', 1,'/',len(robot.positions)+1)
    robot.save_data(feather_folder,name)
    robot.poses.append(name)
    for pose in range(len(robot.positions)):
        name = 'Pose_'+str(pose+1)
        where = robot.positions[pose]
        print('Moving to pose', pose+2,'/',len(robot.positions)+1)
        print('Elevation: ', -1*robot.angles[pose],' [Degrees]')
        if not fake_robot:
            robot.move(where,'pose') # move one step on route
            time.sleep(10)
        print('Taking image')
        robot.take_img(index=(pose+1),fake_camera=fake_camera,exmple_file=feather_folder+'Pose_'+str(pose+1)+'.feather')  
        print('Saving image')                      
        robot.save_data(feather_folder,name)
        robot.poses.append(name)
    
    robot.calc_vec(save=True) #calculate the leaf normal vector
    time.sleep(5)
    robot.covert_normal_to_robotic_pose()
    print('Leaf Normal Vector:', np.round(robot.norm_list[0],2))
    print('Desired Robotic Pose:', robot.robotic_step_2)

    if current_measure:
        robot.move_and_measure() # move to location + orientation and measure the current by the normal vector   



if __name__ == '__main__':
    
    robot = Robot(realtime=realtime,fake_robot=fake_robot,current_measure=current_measure,fake_camera=fake_camera) #
    robot.r_plant = r_plant
    robot.best_route = best_route
    findRealtimeLeafOrientation(robot)






