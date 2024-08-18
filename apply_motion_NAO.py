# coding=utf-8

import csv
import qi
import re
import argparse
import numpy as np
import os

def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print("Directory '%s' created" % directory_path)
    else:
        print("Directory '%s' already exists" % directory_path)

# Set up argument parser
parser = argparse.ArgumentParser(description='Control a robot using motion data.')
parser.add_argument('--ip', type=str, required=True, help='Robot\'s IP address ("localhost" for simulator, "nao.lan" for real robot)')
parser.add_argument('--port', type=int, help='Port number (required if ip is "localhost")')
parser.add_argument('--motionpath', type=str, required=True, help='Path to the motion data folder')
parser.add_argument('--datapath', type=str, required=True, help='Name of the CSV file containing the motion data')

args = parser.parse_args()

ip = args.ip
port = args.port
motionpath = args.motionpath
datapath = args.datapath

# Ensure port is provided if ip is "localhost"
if ip == "localhost" and port is None:
    parser.error('--port is required when ip is "localhost"')

input_csv_file = os.path.join(motionpath, datapath)  # CSV file of the motion to be applied to the robot

path = motionpath
create_directory_if_not_exists(path)

output_csv_file = os.path.join(motionpath, "record_" + datapath)  # CSV file for recording the robot's degrees of freedom
output_csv_fileO = os.path.join(motionpath, "recordO_" + datapath)  # CSV file for recording the robot's degrees of freedom (command)
record_interval = 100000  # Interval for recording the robot's degrees of freedom (unit: microseconds)

if ip == "localhost" :
    app = qi.Application(url="tcp://{}:{}".format(ip, port))  # Specify the port
else:
    app = qi.Application(url="tcp://{}".format(ip))  # Do not specify the port

app.start()
session = app.session

motion = session.service("ALMotion")
posture_service = session.service("ALRobotPosture")
motion.wakeUp()
posture_service.goToPosture("StandInit", 0.5)
motion.wbEnable(True)  # Enable balance maintenance function
motion.wbFootState("Fixed", "Legs")  # Attach legs to the ground
motion.wbEnableBalanceConstraint(True, "Legs")  # Keep the center of gravity in the middle
motion.setCollisionProtectionEnabled("Arms", True)  # Prevent the arms from penetrating the torso

with open(input_csv_file) as f:
    lines = iter(f)
    names = re.split(r',\s*', next(lines).strip())[1:]
    angle_lists = np.loadtxt(lines, delimiter=",", unpack=True).tolist()
    time_lists = [angle_lists[0]] * len(names)
    angle_lists = angle_lists[1:]


########## Code for recording the robot's degrees of freedom at regular intervals
# https://developer.softbankrobotics.com/nao6/naoqi-developer-guide/qi-framework/api-references/python-qi-api-reference/qiasync
angles_lst = []
angles_lstO = []

microseconds = 0
def record_angles():
    angles_lst.append(motion.getAngles("Body", True))
    angles_lstO.append(motion.getAngles("Body", False))

get_angles = qi.PeriodicTask()
get_angles.setCallback(record_angles)
get_angles.setUsPeriod(record_interval)
##########

get_angles.start(True)  # Start recording the robot's degrees of freedom
motion.angleInterpolation(names, angle_lists, time_lists, True)
get_angles.stop()  # Stop recording the robot's degrees of freedom

motion.rest()
motion.wbEnable(False)  # Disable balance maintenance function; always run this after the motion ends

with open(output_csv_file, 'wb') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['microseconds'] + motion.getBodyNames("Body"))
    csv_writer.writerows([i * record_interval] + angles for i, angles in enumerate(angles_lst))

with open(output_csv_fileO, 'wb') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['microseconds'] + motion.getBodyNames("Body"))
    csv_writer.writerows([i * record_interval] + angles for i, angles in enumerate(angles_lstO))
