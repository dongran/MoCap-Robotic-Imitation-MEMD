# coding=utf-8
"""
This script applies motion data to NAO robot.
It can be used with both real robot and simulator.

Author: Dong Ran
"""

import csv
import qi
import re
import argparse
import numpy as np
import os


def create_directory_if_not_exists(directory_path):
    """
    Create directory if it doesn't exist.
    
    Parameters:
    -----------
    directory_path : str
        Path of the directory to create
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print("Directory '%s' created" % directory_path)
    else:
        print("Directory '%s' already exists" % directory_path)


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
    --------
    argparse.Namespace
        Parsed arguments
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Control a robot using motion data.')
    parser.add_argument('--ip', type=str, required=True, 
                       help='Robot\'s IP address ("localhost" for simulator, "nao.lan" for real robot)')
    parser.add_argument('--port', type=int, 
                       help='Port number (required if ip is "localhost")')
    parser.add_argument('--motionpath', type=str, required=True, 
                       help='Path to the motion data folder')
    parser.add_argument('--datapath', type=str, required=True, 
                       help='Name of the CSV file containing the motion data')

    args = parser.parse_args()

    # Ensure port is provided if ip is "localhost"
    if args.ip == "localhost" and args.port is None:
        parser.error('--port is required when ip is "localhost"')
        
    return args


def setup_robot_connection(ip, port=None):
    """
    Set up connection to the robot.
    
    Parameters:
    -----------
    ip : str
        IP address of the robot
    port : int or None
        Port number (required for simulator)
        
    Returns:
    --------
    tuple
        qi.Application, qi.Session
    """
    if ip == "localhost":
        app = qi.Application(url="tcp://{}:{}".format(ip, port))
    else:
        app = qi.Application(url="tcp://{}".format(ip))

    app.start()
    return app, app.session


def initialize_robot(session):
    """
    Initialize robot services and posture.
    
    Parameters:
    -----------
    session : qi.Session
        Session connected to the robot
        
    Returns:
    --------
    tuple
        motion service, posture service
    """
    motion = session.service("ALMotion")
    posture_service = session.service("ALRobotPosture")
    
    motion.wakeUp()
    posture_service.goToPosture("StandInit", 0.5)
    
    # Enable balance maintenance function
    motion.wbEnable(True)
    motion.wbFootState("Fixed", "Legs")  # Attach legs to the ground
    motion.wbEnableBalanceConstraint(True, "Legs")  # Keep the center of gravity in the middle
    motion.setCollisionProtectionEnabled("Arms", True)  # Prevent the arms from penetrating the torso
    
    return motion, posture_service


def load_motion_data(file_path):
    """
    Load motion data from CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing motion data
        
    Returns:
    --------
    tuple
        names, angle_lists, time_lists
    """
    with open(file_path) as f:
        lines = iter(f)
        names = re.split(r',\s*', next(lines).strip())[1:]
        angle_lists = np.loadtxt(lines, delimiter=",", unpack=True).tolist()
        time_lists = [angle_lists[0]] * len(names)
        angle_lists = angle_lists[1:]
    
    return names, angle_lists, time_lists


def record_robot_motion(motion, record_interval=100000):
    """
    Set up a periodic task to record robot's angles.
    
    Parameters:
    -----------
    motion : qi.ALMotion
        Motion service instance
    record_interval : int
        Interval for recording in microseconds
        
    Returns:
    --------
    tuple
        angles_lst, angles_lstO, get_angles task
    """
    angles_lst = []
    angles_lstO = []

    def record_angles():
        angles_lst.append(motion.getAngles("Body", True))
        angles_lstO.append(motion.getAngles("Body", False))

    get_angles = qi.PeriodicTask()
    get_angles.setCallback(record_angles)
    get_angles.setUsPeriod(record_interval)
    
    return angles_lst, angles_lstO, get_angles


def save_recorded_data(output_file, output_fileO, angles_lst, angles_lstO, motion, record_interval):
    """
    Save recorded angles to CSV files.
    
    Parameters:
    -----------
    output_file : str
        Path to save true angles
    output_fileO : str
        Path to save commanded angles
    angles_lst : list
        List of recorded true angles
    angles_lstO : list
        List of recorded commanded angles
    motion : qi.ALMotion
        Motion service instance
    record_interval : int
        Interval used for recording
    """
    with open(output_file, 'wb') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['microseconds'] + motion.getBodyNames("Body"))
        csv_writer.writerows([i * record_interval] + angles for i, angles in enumerate(angles_lst))

    with open(output_fileO, 'wb') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['microseconds'] + motion.getBodyNames("Body"))
        csv_writer.writerows([i * record_interval] + angles for i, angles in enumerate(angles_lstO))


def main():
    """
    Main function to execute the robot motion application.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up paths
    motionpath = args.motionpath
    create_directory_if_not_exists(motionpath)
    
    input_csv_file = os.path.join(motionpath, args.datapath)
    output_csv_file = os.path.join(motionpath, "record_" + args.datapath)
    output_csv_fileO = os.path.join(motionpath, "recordO_" + args.datapath)
    
    # Record interval (microseconds)
    record_interval = 100000
    
    # Connect to the robot
    app, session = setup_robot_connection(args.ip, args.port)
    
    # Initialize robot
    motion, posture_service = initialize_robot(session)
    
    # Load motion data
    names, angle_lists, time_lists = load_motion_data(input_csv_file)
    
    # Set up recording
    angles_lst, angles_lstO, get_angles = record_robot_motion(motion, record_interval)
    
    # Execute motion and record
    get_angles.start(True)
    motion.angleInterpolation(names, angle_lists, time_lists, True)
    get_angles.stop()
    
    # Return to rest position and disable balance functions
    motion.rest()
    motion.wbEnable(False)
    
    # Save recorded data
    save_recorded_data(output_csv_file, output_csv_fileO, angles_lst, angles_lstO, 
                      motion, record_interval)


if __name__ == "__main__":
    main()
