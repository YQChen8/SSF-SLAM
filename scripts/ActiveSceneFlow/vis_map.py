from pickle import FALSE

from matplotlib.markers import MarkerStyle
import carla
import argparse
import matplotlib.pyplot as plt
import pygame
import numpy as np
from sklearn.neighbors import NearestNeighbors

def dist( srcx, srcy, dstx, dsty):
        dist = np.sqrt((srcx-dstx)**2 + (srcy-dsty)**2)
        # print(dist)
        return dist

class RoadPoint:
    '''
    x, y: position
    each road side has two values, left=1, right=-1;
    if left side of road exist, left = 1;
    if right sideof road exist, right = -1;
    '''
    def __init__(self, x, y, z, rd_w, left=None, right=None, junction=False, intersection=False):
        self.x = x
        self.y = y
        self.z = z
        self.road_width = rd_w
        self.left = None
        self.right = None
        self.junction = False
        self.intersection = False
        if left is not None or self.junction or self.intersection:
            self.left = 1
        if right is not None or self.junction or self.intersection:
            self.right = -1
        


class MapVisualization:
    def __init__(self, args):
        self.carla_client = carla.Client(args.host, args.port, worker_threads=1)
        self.world = self.carla_client.get_world()
        self.map = self.world.get_map()
        self.fig, self.ax = plt.subplots()

    def destroy(self):
        self.carla_client = None
        self.world = None
        self.map = None

    @staticmethod
    def lateral_shift(transform, shift):
        """Makes a lateral shift of the forward vector of a transform"""
        transform.rotation.yaw += 90
        return transform.location + shift * transform.get_forward_vector()

    def draw_line(self, points: list, road_types: list):
        x = []
        y = []
        for p in points:
            x.append(p.x)
            y.append(-p.y)
        self.ax.plot(x, y, color='darkslategrey', markersize=2)
        return True

    # def dist(self, srcx, srcy, dstx, dsty):
    #     dist = np.sqrt((srcx-dstx)**2 + (srcy-dsty)**2)
    #     # print(dist)
    #     return dist

    def draw_road_line(self, points: list, color = 'darkslategrey', step=20):
        x = []
        y = []
        x1 = []
        y1 = []
        # for p in points:
  
        for i in range(len(points)):
            if i % step == 0:
                p = points[i]
            # if (p.left is not None and p.right is None) or (p.left is  None and p.right is not None):
                # if p.right is not None:
                #     print('right')
                # if p.left is not None or p.right is not None:
                #     x.append(p.x)
                #     y.append(-p.y)
                # else:
                #     x1.append(p.x)
                #     y1.append(-p.y)
                if p.left is not None:
                    x.append(p.x)
                    y.append(-p.y)
                if p.right is not None:
                    x1.append(p.x)
                    y1.append(-p.y)
        # self.ax.plot(x, y, color=color, markersize=2, marker = '+')
        # self.ax.plot(x1, y1, color='black', markersize=2)
        self.ax.plot(x, y, 'r+')
        self.ax.plot(x1, y1, 'g+')
        return True

    def draw_points(self, points: list, color_id, step=20):
        x = []
        y = []
        for p in points:
        # for i in range(len(points)):
        #     if i % step == 0:
        #         p = points[i][0]
        #         x.append(p.x)
        #         y.append(-p.y)
            x.append(p[0])
            y.append(-p[1])
        if color_id == 0:
            self.ax.plot(x, y, 'b.')
        elif color_id == 1:
            self.ax.plot(x, y, 'r.')
        elif color_id == 2:
            self.ax.plot(x, y, 'g.')
        else:
            self.ax.plot(x, y, 'y.')
        return True
    
    def draw_bbx(self, bbx):
        x = bbx.location.x - bbx.extent.x
        y = bbx.location.y - bbx.extent.y
        width = bbx.extent.x * 2.0
        height = bbx.extent.y * 2.0

        self.ax.add_patch(plt.Rectangle((x,y), width, height))
        return True

    def draw_lane_type(self, points: list, road_types: list):
        x = []
        y = []
        tmp = str(road_types[0])
        for i in range(len(points)):
            p = points[i]
            x=p.x
            y=-p.y
            # if '933' in road_types[0]:
            #     self.ax.text(x, y, tmp,
            #                     fontsize=6)
            if i == 0:
                self.ax.text(x, y, tmp,
                                fontsize=6)
            else:
                if tmp not in str(road_types[i]):
                    self.ax.text(x, y, tmp,
                                    fontsize=6)
            tmp = str(road_types[i])
            

    def draw_spawn_points(self):
        spawn_points = self.map.get_spawn_points()
        for i in range(len(spawn_points)):
            p = spawn_points[i]
            x = p.location.x
            y = -p.location.y
            self.ax.text(x, y, str(i),
                         fontsize=6,
                         color='darkorange',
                         va='center',
                         ha='center',
                         weight='bold')

    def judge_pts_pos(self, center_pt, ref_pt):
        '''
                0 1 2        3 4 5    6          7 8 9   10         11
        center:[x,y,z]; wp1:[x,y,z,road_id];wp2:[x,y,z,road_id];[wp_dist]
        ref_pt: [x,y,z]
        '''
        A = center_pt[3] - center_pt[7]
        B = center_pt[4] - center_pt[8]
        C = center_pt[5] - center_pt[9]

        value = A * (ref_pt[0] - center_pt[0]) + B * (ref_pt[1] - center_pt[1]) + C * (ref_pt[2] - center_pt[2])
        if value < 0:
            shift_w = center_pt[11]
        else:
            shift_w = -center_pt[11]
        value = abs(value) / np.sqrt(A ** 2 + B ** 2 + C ** 2)
        return shift_w, value

    def Comprehension_Waypoints(self, wp, shift_widith, set_road_ids):
        pt = self.lateral_shift(wp.transform, shift_widith) 
        rd_id = 0
        if wp.road_id in set_road_ids[0]:
            rd_id = 0
        elif wp.road_id in set_road_ids[1]:
            rd_id = 1
        elif wp.road_id in set_road_ids[2]:
            rd_id = 2
        else :
            rd_id = 3
        return [pt.x,pt.y,pt.z, rd_id]

    def draw_roads(self):
        precision = 0.1
        topology = self.map.get_topology()
        topology = [x[0] for x in topology]
        topology = sorted(topology, key=lambda w: w.transform.location.z)
        set_waypoints = []
        skeletions = []
  
        for waypoint in topology:
            waypoints = [waypoint]
            skeletions += [[waypoint.transform.location.x,waypoint.transform.location.y,waypoint.transform.location.z,waypoint.road_id]]
            nxt = waypoint.next(precision)
            if len(nxt) > 0:
                nxt = nxt[0]
                while nxt.road_id == waypoint.road_id:
                    waypoints.append(nxt)
                    nxt = nxt.next(precision)
                    if len(nxt) > 0:
                        nxt = nxt[0]
                    else:
                        break
            set_waypoints.append(waypoints)

        # np_b = np.array(skeletions)
        # np.save('./np_s', np_b)
        
        road_left_side = []
        road_right_side = []
        set_left_waypoints = [[],[],[],[]]
        set_right_waypoints = [[],[],[],[]]
        
        # set_road_ids = [set([6,89,125,735,801]), set([0,3,7,10,17,90]), set([5,516,763,795]), set([1,4,8,515,675])]
        set_road_ids = [set([1,3,12,15,16]),set([0,2,18,300]),set([5]),set([0,2,13,17,18,19,300])]

        xx = []
        yy = []
        sub_skeletions = []
        iou_s_ids = []
        for i, item in enumerate(skeletions):
            if item[3] in set_road_ids[0] or item[3] in set_road_ids[1] or item[3] in set_road_ids[2] or item[3] in set_road_ids[3]:
                xx.append(item[0])
                yy.append(-item[1])
                sub_skeletions += [[item[0], item[1], item[2]]]
                iou_s_ids += [i]
        self.ax.plot(xx, yy, 'r.')
        skeletions = np.array(skeletions)
        sub_skeletions = np.array(sub_skeletions)
        n_sample = 5
        nnbrs = NearestNeighbors(n_neighbors=n_sample, algorithm='ball_tree').fit(skeletions[:,:3])
        distances, indices = nnbrs.kneighbors(skeletions[:,:3])
        print(indices[iou_s_ids,:])
        print('*'*20)
        print(distances[iou_s_ids,:])
        distances = distances[iou_s_ids,:]
        indices = indices[iou_s_ids,:]
        # sub_skeletions = sub_skeletions[iou_s_ids,:]
        pair_inds = []
        pair_dists = []
        center_waypoints = []
        xxx = []
        yyy = []
        for i, item in enumerate(distances):
            for j in range(1,n_sample,1):
                if item[j] >= 0.05:
                    pair_inds += [[indices[i,0], indices[i,j]]]
                    pair_dists += [[distances[i,0], distances[i,j]]]
                    tmp = (skeletions[indices[i,0],:3] + skeletions[indices[i,j],:3]) / 2.0
                    center_dist = np.linalg.norm(skeletions[indices[i,0],:3] - np.array(tmp))
                    center_waypoints += [list(tmp)+list(skeletions[indices[i,0],:]) + list(skeletions[indices[i,j],:]) + [center_dist]]
                    xxx += [tmp[0]]
                    yyy += [-tmp[1]]
                    break
        print(pair_inds)
        print('*'*20)
        print(pair_dists)

        self.ax.plot(xxx, yyy, 'g.')

    
        i_CenterWpts = 0
        all_set_center_waypoints = []
        set_road_width = []
        rd_w = None
        for i, waypoints in enumerate(set_waypoints):
            road_left_side = []
            road_right_side = []
            # waypoint = waypoints[0]
            road_wps = [self.lateral_shift(w.transform, 0) for w in waypoints]
            # road_left_side = [self.lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints]
            # road_right_side = [self.lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints]
            if i in iou_s_ids:
                shift_w, _ = self.judge_pts_pos(center_waypoints[i_CenterWpts], [road_wps[0].x, road_wps[1].y, road_wps[2].z])
                set_center_waypoints = [self.Comprehension_Waypoints(w, shift_w, set_road_ids) for w in waypoints]
                i_CenterWpts += 1
            
            for iw, w in enumerate(waypoints):
                left_pos = self.lateral_shift(w.transform, -w.lane_width * 0.5)
                right_pos = self.lateral_shift(w.transform, w.lane_width * 0.5)
                

                left = None
                right = None
                l = w.get_left_lane()
                is_exist = False
                is_left = True
                while l and l.lane_type != carla.LaneType.Driving:
                    i = 0
                    is_exist = False
                    # if l.left_lane_marking.type==carla.LaneMarkingType.Curb or l.right_lane_marking.type==carla.LaneMarkingType.Curb:
                    if l.lane_type == carla.LaneType.Shoulder or l.lane_type == carla.LaneType.Sidewalk or l.lane_type == carla.LaneType.Median:
                        if not l.is_junction:
                            if l.road_id in set_road_ids[0]:
                                is_exist = True
                                i = 0
                                if is_left:
                                    is_left = True
                                    left = 1
                                else:
                                    is_left = False
                                    right = -1
                            elif l.road_id in set_road_ids[1]:
                                is_exist = True
                                i = 1
                                if is_left:
                                    is_left = True
                                    left = 1
                                else:
                                    is_left = False
                                    right = -1
                            elif l.road_id in set_road_ids[2]:
                                is_exist = True
                                i = 2
                                if is_left:
                                    is_left = True
                                    left = 1
                                else:
                                    is_left = False
                                    right = -1
                            elif l.road_id in set_road_ids[3]:
                                is_exist = True
                                i = 3
                                if is_left:
                                    is_left = True
                                    left = 1
                                else:
                                    is_left = False
                                    right = -1
                            if i_CenterWpts > 0:
                                rd_w = np.linalg.norm((np.array(set_center_waypoints[iw][:3]) - np.array([left_pos.x, left_pos.y, left_pos.z])))

                    
                    l = l.get_left_lane()
                    is_left = False
                
                
                pos = RoadPoint(left_pos.x, left_pos.y, left_pos.z, rd_w, left=left, right=right)
                rd_w = None
                if is_exist:
                    set_left_waypoints[i] += [pos]
                    is_exist = False
                road_left_side += [pos]

                left = None
                right = None
                r = w.get_right_lane()

                is_exist = False
                is_left = True
                while r and r.lane_type != carla.LaneType.Driving:
                    i = 0
                    is_exist = False
                    if r.lane_type == carla.LaneType.Shoulder or r.lane_type == carla.LaneType.Sidewalk or r.lane_type == carla.LaneType.Median:
                        if not r.is_junction:
                            if r.road_id in set_road_ids[0]:
                                i = 0
                                is_exist = True
                                # if is_left:
                                if r.lane_id > 0:
                                    is_left = True
                                    left = 1
                                else:
                                    is_left = False
                                    right = -1
                            elif r.road_id in set_road_ids[1]:
                                i = 1
                                is_exist = True
                                if r.lane_id > 0:
                                    is_left = True
                                    left = 1
                                else:
                                    is_left = False
                                    right = -1
                            elif r.road_id in set_road_ids[2]:
                                i = 2
                                is_exist = True
                                if r.lane_id > 0:
                                    is_left = True
                                    left = 1
                                else:
                                    is_left = False
                                    right = -1
                            elif r.road_id in set_road_ids[3]:
                                i = 3
                                is_exist = True
                                if r.lane_id > 0:
                                    is_left = True
                                    left = 1
                                else:
                                    is_left = False
                                    right = -1
                            if i_CenterWpts > 0 and is_exist:
                                rd_w = np.linalg.norm((np.array(set_center_waypoints[iw][:3]) - np.array([right_pos.x, right_pos.y, right_pos.z])))

                    r = r.get_right_lane()

                
                pos = RoadPoint(right_pos.x, right_pos.y, right_pos.z, rd_w, left=left, right=right)
                rd_w = None
                if is_exist:
                    set_right_waypoints[i] += [pos]
                    is_exist = False
                road_right_side += [pos]

                # if left is not None or right is not None:

            # road_left_sides += [road_left_side]
            # road_right_sides += [road_right_side]
            # road_left_type = [str(w.left_lane_marking.type) + ' ' +  str(w.right_lane_marking.type) for w in waypoints]
            # road_right_type = [str(w.left_lane_marking.type) + ' ' +  str(w.right_lane_marking.type) for w in waypoints]
            road_left_type = [str(w.road_id) for w in waypoints]
            # road_left_type = [str(w.road_id) for w in waypoints]
            # self.draw_lane_type(points=road_wps, road_types=road_left_type)
            # self.draw_points(points=wp_junc, road_types=road_left_type)
            # self.draw_points(points=road_wps, road_types=road_left_type)
            # plt.axis('equal')
            # plt.show()
            
                
        
            if len(road_left_side) > 2:
                # self.draw_points(points=wp_junc, road_types=road_left_type)
                # self.draw_points(points=road_wps,color_id=4)
                if i_CenterWpts > 0:
                    # tmp_set_center_waypoints = set_center_waypoints
                    # set_center_waypoints = []
                    # for i, item in enumerate(set_center_waypoints):
                    #     np_set_center_waypoints += [[item[0].x, item[0].y, item[0].z, item[1]]]
                    self.draw_points(points=set_center_waypoints,color_id=3)
                    
                self.draw_road_line(points=road_left_side)
                if len(road_left_type) > 0:
                    self.draw_lane_type(points=road_wps, road_types=road_left_type)
                
            if len(road_right_side) > 2:
                self.draw_road_line(points=road_right_side)
                # if len(road_right_type) > 0:
                #     self.draw_lane_type(points=road_wps, road_types=road_right_type)
            # if w.is_junction:
            #     plt.axis('equal')
            #     plt.show()
            if i_CenterWpts > 0:
                all_set_center_waypoints += set_center_waypoints
        return set_left_waypoints, set_right_waypoints, all_set_center_waypoints


def draw_roadline(ax, points: list, color = 'darkslategrey', step=2, is_center=False):
        x = []
        y = []
        x1 = []
        y1 = []
        # for p in points:
        if not is_center:
            tmp_x =  points[0].x
            tmp_y =  -points[0].y
        for i in range(len(points)):
            if is_center:
                x.append(points[i,0])
                y.append(-points[i,1])
            else:
                if i % step == 0:
                    p = points[i]
                # if (p.left is not None and p.right is None) or (p.left is  None and p.right is not None):
                    if p.left is not None or p.right is not None:
                        if dist(tmp_x, tmp_y, p.x, -p.y) < 10.0:
                            x.append(p.x)
                            y.append(-p.y)
                    else:
                        x1.append(p.x)
                        y1.append(-p.y)
                    tmp_x =  p.x
                    tmp_y =  -p.y
        
        if not is_center:
            ax.plot(x, y, 'g.')
        else:
            ax.plot(x, y, 'r.')
        
        # # plt.draw()#注意此函数需要调用
        # plt.axis('equal')
        #     # time.sleep(0.01)
        # xy = np.array([x,y])
        # new_xy = np.sort(xy.T,axis=0)
        # # print(xy.shape)
        #     # plt.show()
        # ax.plot(new_xy[:,0], new_xy[:,1], color='green', markersize=2)
        # plt.axis('equal')
        # plt.show()
        # print(xy.shape)
        return True


from matplotlib.patches import Polygon
import numpy as np
def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host CARLA Simulator (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port of CARLA Simulator (default: 2000)')
    argparser.add_argument(
        '-m', '--map',
        default='Town02',
        help='Load a new map to visualize'
    )

    args = argparser.parse_args()
    viz = MapVisualization(args)
    a,b,c=viz.draw_roads()
    # viz.draw_lane_type()
    # viz.draw_spawn_points()
    viz.destroy()
    plt.axis('equal')
    plt.show()
    # return 
    color = ['red', 'green', 'blue', 'darkslategrey']
    # if a:
    #     for i in len(a):
    #         viz.draw_road_line(a[i], color[i])
    np_b = np.array(b)
    np.save('./np_b', np_b)
    np_c = np.array(c)
    # np_c[:,1] *= -1.0
    np_c[:,2] = -2.5
    np.save('./np_c', np_c)
    # if b:
    # # b = np.load('./np_b.npy', allow_pickle=True)
    # # if b.all:
    #     for i in range(len(b)):
    #         # draw_roadline(b[i], color[i])
    #         # viz.draw_points(b[i], i)
    #         viz.draw_road_line(b[i], color[i])
    #     plt.axis('equal')
    #     plt.show()
   

def save_np_data(input, filename, z=-2.5, is_road_center=False):
    pts = []
    if input.all:
        # tmp_pts = []
        # clr_pts = []
        # cnt = 0
        if is_road_center:
            pointcloud = input.astype(np.dtype('f4'))
        else:
            for i in range(len(input)):
                # tmp_pts = []
                # cnt = cnt + 1
                for item in input[i]:
                    pts += [[item.x, -item.y, z, i]]
            # pts += [tmp_pts]
        
            pointcloud = np.array(pts, dtype=np.dtype('f4'))
        # points_R = np.exp(-0.05*np.sqrt(np.sum(pointcloud**2,axis=1))).reshape(-1,1)
        # pointcloud = np.concatenate((pointcloud,points_R),axis=1)
        # import os
        # print(os.path.abspath(os.curdir))
        pointcloud.tofile(filename)

if __name__ == "__main__":
    # execute only if run as a script
    # s = np.load('./np_s.npy', allow_pickle=True)
    # x = []
    # y = []
    # fig, ax = plt.subplots()
    # for i in s:
    #     x.append(i[0])
    #     y.append(-i[1])
    # ax.plot(x, y, 'r+')
    # plt.show()
    main()

    fig, ax = plt.subplots()

    b = np.load('./np_b.npy', allow_pickle=True)
    save_np_data(b,'town02-map.bin')
    for i in range(len(b)):
        draw_roadline(ax,b[i])

    c = np.load('./np_c.npy', allow_pickle=True)
    save_np_data(c,'town02-road-center.bin', is_road_center=True)
    draw_roadline(ax, c,color='red', is_center=True)
    # color = [[0,0,1], [0,1,0], [1,0,0], [1,1,0]]
    # x = []
    # y = []
    # fig, ax = plt.subplots()
    # for i in c:
    #     x.append(i[0])
    #     y.append(-i[1])
    # ax.plot(x, y, 'r+')
    plt.show()
