# Motion Planning of Mbot Mega Robot on Qualcomm RB5 Platform

<p align="justify">
This project focuses on implementation of search based path planning algorithm with PID control system on Mbot Mega robot powered using Qualcomm RB5 platform. The goal of the project is to enable the robot reach the target position following a shortest path in a 10ft ×10ft area with visual feedback from April tag landmarks. The project further address two specific requirements about the nature of the path traversed in terms of distance optimality and maximum safety.
</p>

## Project Report
[Orish Jindal, Sanchit Gupta, 'Motion Planning of Mbot Mega Robot on Qualcomm RB5 Platform', CSE 276A, Course Project, UCSD](https://github.com/sanchit3103/Motion-Planning_Mbot-Mega-with-RB5/blob/main/Report.pdf)

## Robot in action: Shortest Path 
<p align="center">
  <img src = "https://user-images.githubusercontent.com/4907348/208624833-65c221fe-e0d4-43a0-8dcd-3e47524f2cb8.gif"/>
 </p>
 
 ## Robot in action: Safest Path
<p align="center">
  <img src = "https://user-images.githubusercontent.com/4907348/208625429-2e302156-168f-4811-8213-295a733b27d2.gif"/>
</p>

## Details to run the code

The architecture of the code includes the following four nodes:

* <b> Node 1: </b> Camera node initialized using rb5_camera_main_ocv.launch executable
* <b> Node 2: </b> April tag detection node initialized using apriltag_detection_array.py executable
* <b> Node 3: </b> MPI control node initialized using hw2_mpi_control_node.py executable
* <b> Node 4: </b> Path planning node initialized using hw4_sol.py executable
