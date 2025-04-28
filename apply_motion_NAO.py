# coding=utf-8
"""
This script applies motion data to NAO robot.
It can be used with both real robot and simulator.

Author: Dong Ran
"""

import os
from utils.robot import (
    create_directory_if_not_exists,
    parse_arguments,
    setup_robot_connection,
    initialize_robot,
    load_motion_data,
    record_robot_motion,
    save_recorded_data
)


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
