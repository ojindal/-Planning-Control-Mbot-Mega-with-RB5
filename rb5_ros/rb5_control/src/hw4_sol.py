#!/usr/bin/env python
import sys

# from pyrsistent import T
import rospy
from geometry_msgs.msg import Twist
from april_detection.msg import AprilTagDetectionArray
from geometry_msgs.msg import Pose2D
import numpy as np
import time
import math
import csv
scale = 0.3
"""
The class of the pid controller.
"""

# Store the world frame coordinates of landmarks in this dictionary
AprilPoseDict = {}

AprilPoseDict[7] = [(3.048,2.8,0),(1.53,1.3,-np.pi/2)]
AprilPoseDict[8] = [(2.8,0,np.pi/2), (2.8,3.048,-np.pi/2)]
AprilPoseDict[1] = [(3.048,0.2,0), (1.7,1.52,np.pi)]
AprilPoseDict[11] = (1.53,1.74,np.pi/2)
AprilPoseDict[10] = (1.36,1.52,0)

AprilPoseDict[4] = (0.2,0,np.pi/2)
AprilPoseDict[3] = (0.2,3.048,-np.pi/2)
AprilPoseDict[5] = [(0,2.8,-np.pi), (0,0.2,-np.pi)]

class PIDcontroller:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.target = np.array([0.0,0.0,0.0])
        self.I = np.array([0.0,0.0,0.0])
        self.lastError = np.array([0.0,0.0,0.0])
        self.timestep = 0.1
        self.maximumValue = 0.1

    def setTarget(self, targetx, targety, targetw):
        """
        set the target pose.
        """
        # self.I = np.array([0.0,0.0,0.0])
        # self.lastError = np.array([0.0,0.0,0.0])
        self.target = np.array([targetx, targety, targetw])

    def setTarget(self, state):
        """
        set the target pose.
        """
        # self.I = np.array([0.0,0.0,0.0])
        # self.lastError = np.array([0.0,0.0,0.0])
        self.target = np.array(state)

    def getError(self, currentState, targetState):
        """
        return the difference between two states
        """
        result = targetState - currentState
        result[2] = (result[2] + np.pi) % (2 * np.pi) - np.pi
        return result

    def setMaximumUpdate(self, mv):
        """
        set maximum velocity for stability.
        """
        self.maximumValue = mv

    def update(self, currentState):
        """
        calculate the update value on the state based on the error between current state and target state with PID.
        """
        e = self.getError(currentState, self.target)

        P = self.Kp * e
        self.I = self.I + self.Ki * e * self.timestep
        I = self.I
        D = self.Kd * (e - self.lastError)
        result = P + I + D

        self.lastError = e

        # scale down the twist if its norm is more than the maximum value.
        resultNorm = np.linalg.norm(result)
        if(resultNorm > self.maximumValue):
            result = (result / resultNorm) * self.maximumValue
            self.I = 0.0

        return result

def genTwistMsg(desired_twist):
    """
    Convert the twist to twist msg.
    """
    twist_msg = Twist()
    twist_msg.linear.x  = desired_twist[0]
    twist_msg.linear.y  = desired_twist[1]
    twist_msg.linear.z  = 0
    twist_msg.angular.x = 0
    twist_msg.angular.y = 0
    twist_msg.angular.z = desired_twist[2]
    return twist_msg

def coord(twist, current_state):
    J = np.array([[np.cos(current_state[2]), -np.sin(current_state[2]), 0.0],
                  [np.sin(current_state[2]), np.cos(current_state[2]), 0.0],
                  [0.0,0.0,1.0]])
    return np.dot(J, twist)

def euler_from_quaternion(x, y, z, w):

    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z # in radians


def camera_call(camera_cmd):

    if len(camera_cmd.detections) != 0:

        global wf_robot_pos

        t_pos   = np.array([0.0,0.0,0.0])
        div     = 0
        for tag in camera_cmd.detections:
            if int(tag.id) in AprilPoseDict.keys():
                id      = int(tag.id)
                x1      = tag.pose.position.z
                y1      = tag.pose.position.x
                qx      = tag.pose.orientation.x
                qy      = tag.pose.orientation.y
                qz      = tag.pose.orientation.z
                qw      = tag.pose.orientation.w
                y_about_x, z_about_y, x_about_z     = euler_from_quaternion(qx, qy, qz, qw)
                th1     = -z_about_y

                x0, y0, th0     = AprilPoseDict[id]
                # print('tag coords', np.array([x1,y1,th1]))
                th2     = th0 - th1
                th2     = (th2 + np.pi) % (2 * np.pi) - np.pi
                dx      = x1*np.cos(th2) + y1*np.sin(th2)
                dy      = y1*np.cos(th2) + x1*np.cos(np.pi/2 + th2)

                pos     = np.array([x0 - dx, y0 - dy, th2])
                # print('pos', pos)
                t_pos   += pos
                div     += 1
        if div:
            wf_robot_pos    = t_pos/div
        # print('Robot pos in cam call: ', wf_robot_pos)


AprilPoseDict = {}
AprilPoseDict[7] = (3.048,2.8,0)
AprilPoseDict[8] = (2.8,0,np.pi/2)
AprilPoseDict[1] = (3.048,0.2,0)

AprilPoseDict[11] = (1.53,1.74,np.pi/2)
AprilPoseDict[10] = (1.36,1.52,0)

def navigate(_current_state, _target_state):

    if not np.array_equal(_current_state, _target_state):

        _current_state  = np.float32(_current_state)
        pid.setTarget(_target_state)

        # calculate the current twist
        update_value    = pid.update(_current_state)

        # publish the twist
        pub_twist.publish(genTwistMsg(coord(update_value, _current_state)))

        time.sleep(0.1)

        # update the current state
        _current_state = _current_state + update_value
        plot_path.append(_current_state)

        global wf_robot_pos
        wf_robot_pos    = np.array([0.0, 0.0, 0.0])
        _previous_state = np.array([0.0, 0.0, 0.0])
        rospy.Subscriber('/apriltag_detection_array', AprilTagDetectionArray, camera_call, queue_size=1)

        while(np.linalg.norm(pid.getError(_current_state, _target_state)) > 0.2): # check the error between current state and current way point
            print('Error norm: ', np.linalg.norm(pid.getError(_current_state, _target_state)))

            # Check for robot position in world frame w.r.t landmarks (April Tags)
            # global wf_robot_pos
            # wf_robot_pos    = np.array([0.0, 0.0, 0.0])

            # Subscribe to AprilTagDetectionArray node for updated robot position in world frame
            # rospy.Subscriber('/apriltag_detection_array', AprilTagDetectionArray, camera_call, queue_size=1)
            # rospy.sleep(0.1)

            # print('robot pos in wf: ', wf_robot_pos)

            # Update _current_state with actual position of robot in world frame
            if not np.array_equal(wf_robot_pos, _current_state) and not np.array_equal(wf_robot_pos, np.array([0.0, 0.0, 0.0])):
                if not np.array_equal(_previous_state, wf_robot_pos):
                    print('update robot pos in if cond: ', wf_robot_pos)
                    _previous_state = wf_robot_pos
                    _current_state  = wf_robot_pos

            # calculate the current twist
            update_value    = pid.update(_current_state)

            plot_path.append(_current_state)

            # publish the twist
            pub_twist.publish(genTwistMsg(coord(update_value, _current_state)))
            time.sleep(0.05)

            # update the current state
            _current_state  = _current_state + update_value

            time.sleep(0.05)
    else:
        pub_twist.publish(genTwistMsg(np.array([0.0,0.0,0.0])))

def dis_b(node):
    x, y = node[0], node[1]
    return min(abs(95-x), abs(95-y), abs(0-x), abs(0-y))

def dis_o(node):
    x, y = node[0], node[1]
    return min(abs(58-x), abs(58-y), abs(42-x), abs(42-y))

# A function that gives possible childrens for a node (full env)
def children(node, env):

    # Directions (8-connected grid)
    move = [[0,1],[1,1],[1,0],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]]

    # rbound,cbound = env.shape[0]-1, env.shape[1]-1
    node = np.array(node)
    ans = []
    for i in move:
        look = np.array(i)
        step = node + look
        r,c = step[0], step[1]
        if env[r,c] == 0:
            ans.append(step)
    return ans # array of np.arrays

# Heuristics (euclidean distance 2D - shorter because assumes no obstacles)
def h(node, goal):
    node = np.array(node)
    goal = np.array(goal)
    heur = np.linalg.norm(node-goal)
    return heur

parent = {}

def Astar(robotpos, targetpos, envenv):

    '''
    Fills the array next_states.

    arguments: number of nodes to expand 'n', start node 'robotpos', current goal position 'targetpos', env 'envenv'
    '''
    ulta_path = []

    OPEN, CLOSED = [], []
    OPEN.append((0 + h(robotpos, targetpos) ,robotpos))
    g = {tuple(robotpos):0}
    f = {tuple(robotpos): 0 + h(robotpos, targetpos)}


    while targetpos not in CLOSED:

        fi, i = OPEN.pop()
        CLOSED.append(tuple(i))
        for j in children(i, envenv):
            if tuple(j) not in CLOSED:
                if tuple(j) not in g or g[tuple(j)] > g[tuple(i)] + 1 + env2[j[0], j[1]]:
                    if tuple(j) in g:
                        temp = g[tuple(j)]
                        if (temp + h(temp, targetpos),j) in OPEN:
                            OPEN.remove((temp + h(temp, targetpos),j))
                    g[tuple(j)] = g[tuple(i)] + 1 + + env2[j[0], j[1]]
                    parent[tuple(j)] = tuple(i)
                    f[tuple(j)] = g[tuple(j)] + h(j,targetpos)

                    OPEN.append((f[tuple(j)],j))
                    OPEN.sort(key = lambda x:x[0])
                    OPEN = OPEN[::-1]
    scale = 0.3048*(10/100)
    path_m = [np.array(targetpos)*scale]
    ulta_path = [tuple(targetpos)]
    while tuple(robotpos) not in ulta_path:
        # print('okay')
        node = ulta_path[-1]
        pnode = parent[node]

        ulta_path.append(tuple(pnode))
        path_m.append(np.array(pnode)*scale)

    return ulta_path[::-1], path_m[::-1]

def suitable_path(path_m):
        '''
        -Takes 2D path.
        -Converts to 3D and removes intermediate waypoints.
        '''
        path_3d = []
        # print(path_m)
        for i in range(len(path_m)-1):
            x0, x1 = path_m[i][0], path_m[i+1][0]
            y0, y1 = path_m[i][1], path_m[i+1][1]
            theta = 0
            if x1-x0:
                theta = np.arctan((y0-y1)/(x1-x0))
            wp = np.array([x0,y0,theta])
            path_3d.append(wp)
        path_3d.append(np.array([x1,y1,theta]))

        final_path = [path_3d[0]]
        for i in path_3d:
            if (i[2]*100)//1 != (100*final_path[-1][2])//1:
                # i[2] = (i[2]*100)//1
                final_path.append(i)
        return final_path

if __name__ == "__main__":

    rospy.init_node("hw4")
    pub_twist           = rospy.Publisher("/twist", Twist, queue_size=1)

    env                 = np.zeros((100,100))

    # Car safety 0.5 feet
    # obs 1.5x1.5 feet

    env[0:5,:]          = 1
    env[:,0:5]          = 1
    env[-5:-1,:]        = 1
    env[:,-5:-1]        = 1

    obs                 = np.ones((20,20))
    env[40:60, 40:60]   = obs

    # a duplicate environment containings costs of each node in original environment for A*
    env2 = np.zeros((100,100))

    for i in range(100):
        for j in range(100):
            node = [i,j]
            if dis_o(node) and dis_b(node):
                # Inverse of distance from obstacle added to inverse of dis. from boundary 
                env2[i,j] = (1/dis_b(node)) + (1/dis_o(node))

    robotpos            = 10,90
    targetpos           = 90,10

    _, path_m           = Astar(robotpos, targetpos, env)

    plot_path           = []

    # for the safest path
    waypoints = suitable_path(path_m)
    
    # For the shortest path
    # waypoints = [np.array([0.2, 2.9,  0.5404195]), np.array([1.82   , 1.82   , 0.5404195]), np.array([2.9    , 0.2    , 1.03037683])]
    
    for i in range(len(waypoints)-1):
        # init pid controller
        pid             = PIDcontroller(0.08,0.008,0.35)
        pid.I           = np.array([0.0,0.0,0.0])
        pid.lastError   = np.array([0.0,0.0,0.0])
        current_state   = waypoints[i]
        target_state    = waypoints[i+1]
        print('Current state: {} and target state: {}'.format(current_state, target_state))
        plot_path.append(current_state)
        navigate(current_state, target_state)
    
    filename = 'plot_path_safest.csv'
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(plot_path)

    pub_twist.publish(genTwistMsg(np.array([0.0,0.0,0.0])))
