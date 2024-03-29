# !/usr/bin/env python3

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import argparse


class CAVcontrol_Thread(object):   
    #   继承父类threading.Thread
    def __init__(self, vehicle, world, destination, num_min_waypoints, control):
        object.__init__(self)
        self.v = vehicle
        self.w = world
        self.d = destination
        self.n = num_min_waypoints
        self.c = control
        # self.start()   

    def run(self):                      
            #   把要执行的代码写到run函数里面 线程在创建后会直接运行run函数 
            self.control = None
            self.v.update_information(self.w)
            if len(self.v.get_local_planner().waypoints_queue) < self.n:
                self.v.reroute(self.d)
            speed_limit = self.v.vehicle.get_speed_limit()
            self.v.get_local_planner().set_speed(speed_limit)
            self.control = self.c(self.v.vehicle.id, self.v.run_step())


def main():

    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        # default='127.0.0.1',
        default='10.16.75.94',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-s', '--start',
        metavar='S',
        default=0.0,
        type=float,
        help='starting time (default: 0.0)')
    argparser.add_argument(
        '-d', '--duration',
        metavar='D',
        default=0.0,
        type=float,
        help='duration (default: 0.0)')
    argparser.add_argument(
        '-f', '--recorder-filename',
        metavar='F',
        default="test1.log",
        help='recorder filename (test1.log)')
    argparser.add_argument(
        '-c', '--camera',
        metavar='C',
        default=0,
        type=int,
        help='camera follows an actor (ex: 82)')
    argparser.add_argument(
        '-x', '--time-factor',
        metavar='X',
        default=1.0,
        type=float,
        help='time factor (default 1.0)')
    argparser.add_argument(
        '-i', '--ignore-hero',
        action='store_true',
        help='ignore hero vehicles')
    argparser.add_argument(
        '--spawn-sensors',
        action='store_true',
        help='spawn sensors in the replayed world')
    args = argparser.parse_args()

    try:

        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)

        # set the time factor for the replayer
        client.set_replayer_time_factor(args.time_factor)

        # set to ignore the hero vehicles or not
        client.set_replayer_ignore_hero(args.ignore_hero)

        # replay the session
        print(client.replay_file(args.recorder_filename, args.start, args.duration, args.camera, args.spawn_sensors))
        # print(client.show_recorder_file_info(args.recorder_filename, True))
        recorder_str = client.show_recorder_file_info(args.recorder_filename, True)
       
        with open('logfile-Town02Random.txt', 'w') as f:
            # recorder_str1 = bytes(recorder_str, encoding = "utf8")
            f.write(recorder_str)
        print(recorder_str)


    finally:
        pass


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')


        
