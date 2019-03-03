'''
Usage:
    python visualization.py /tmp LidarVeloBumperFL

Player controls:
    arrow down: start / stop
    arrow left / right: step one frame back / forward
    arrow up: change playback direction
    numblock + / -: faster / slower
    a / b: set loop points
    r: disable / enable looping
'''
from h5py import File
import rospy
import pygame
from pygame.locals import *
import os
import numpy as np

from visualization.visualizer import Visualizer

pygame.init()
pygame.display.set_mode((400, 400))


# pygame.event.set_grab(True)


def main(input_folder, prefix):
    # find all the files to display
    file_list = sorted(os.listdir(input_folder))
    file_list = [os.path.join(input_folder, f) for f in file_list if f.endswith(".hdf5")]
    if prefix is not None:
        file_list = [f for f in file_list if prefix in f]

    # find the way to call the data inside the files
    with File(file_list[0]) as f:
        keys = list(f.keys())
        print(keys)
    for key in keys:
        if "point_cloud" in key:
            pcl_key = key
        if "box" in key:
            bbx_key = key
    semantic_key = "label_id"

    # set up ROS publishing mechanism
    visualizer = Visualizer(pcloud_topic="pcloud",
                            bbox_topic="bboxes")
    rate = rospy.Rate(40)

    # set up player state variables
    n = 0
    incr = 1
    direction = 1
    repeat = False

    while not rospy.is_shutdown():
        if incr != 0:
            try:
                print(file_list[n])
            except IndexError:
                n = 0
                print(file_list[n])

        # load data and publish it
        with File(file_list[n]) as f:
            pcloud = np.array(f[pcl_key])
            bboxes = f[bbx_key]
            semantic_data = np.array(f[semantic_key])
            visualizer.publish(bboxes=bboxes, pcloud=pcloud, pcloud_color=semantic_data)

        # player key press event handling
        for event in pygame.event.get():
            if event.type == QUIT:
                quit()
            # try:
            #     print event.dict["key"]
            # except:
            #     pass
            if event.type == KEYDOWN:
                key = event.dict['key']

                # reverse playback direction
                if key == 273:  # arrow up
                    incr = -incr
                    direction = -direction
                    break

                # step one frame back
                if key == 276:  # arrow left
                    n -= 1
                    break

                # start / stop
                if key == 274:  # arrow down
                    incr = abs(abs(incr) - 1) * direction
                    break

                # step one frame forward
                if key == 275:  # arrow right
                    n += 1
                    break

                # slower
                if key == 269:  # numblock -
                    rate.sleep_dur *= 1.2

                # faster
                if key == 270:  # numblock +
                    rate.sleep_dur /= 1.2

                # set start of playback loop
                if key == 97:  # A
                    repeat_a = n
                    break

                # set end of playback loop
                if key == 98:  # B
                    repeat_b = n
                    n = repeat_a - 4
                    repeat = True
                    break

                # disable / enable looping A -> B
                if key == 114:  # R
                    repeat = not repeat

        pygame.event.pump()

        n += incr  # * 4

        if repeat:
            if n > repeat_b:
                n = repeat_a

        rate.sleep()


########################################################################################################################

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")
    parser.add_argument("prefix", nargs="?")
    args = parser.parse_args()
    main(args.input_folder, args.prefix)
