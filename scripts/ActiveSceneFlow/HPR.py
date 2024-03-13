# import open3d as o3d
# from open3d import *
import numpy as np

# from pyhull.delaunay import DelaunayTri
# from pyhull.voronoi import VoronoiTess
from pyhull.convex_hull import ConvexHull
from pyhull.simplex import Simplex


#
# def draw(vis, vis_list, use_mesh_frame=True, param_file='test.json', save_fov=False):
#     mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 1.4])
#     if vis is None:
#         vis = o3d.visualization.Visualizer()
#         vis.create_window()
#         # ctr = vis.get_view_control()
#         # ctr.set_zoom(0.5)
#         vis.get_render_option().light_on = True
#         vis.get_render_option().point_size = 3.0
#         vis.get_render_option().background_color = [1, 1, 1]
#         # vis.get_render_option().background_color = [0, 0, 0]
#     if use_mesh_frame:
#         mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 1.4])
#         vis.add_geometry(mesh_frame)
#
#     for item in vis_list:
#         vis.add_geometry(item)
#
#     param = o3d.io.read_pinhole_camera_parameters(param_file)
#     ctr = vis.get_view_control()
#     ctr.convert_from_pinhole_camera_parameters(param)
#     vis.run()
#     vis.clear_geometries()
#
#     if save_fov:
#         param = vis.get_view_control().convert_to_pinhole_camera_parameters()
#         o3d.io.write_pinhole_camera_parameters('test.json', param)

def HPR(p,C,param):
    '''
    % p - NxD D dimensional point cloud.
    % C - 1xD D dimensional viewpoint.
    % param - parameter for the algorithm. Indirectly sets the radius.
    %
    % Output:
    % visiblePtInds - indices of p that are visible from C.
    %
    % This code is adapted form the matlab code written by Sagi Katz
    % For more information, see "Direct Visibility of Point Sets", Katz S., Tal
    % A. and Basri R., SIGGRAPH 2007, ACM Transactions on Graphics, Volume 26, Issue 3, August 2007.
    % This method is patent pending.
    '''
    numPts, dim= p.shape
    # Move C to the origin
    p=p-np.tile(C, (numPts, 1))
    # Calculate ||p||
    normp=np.linalg.norm(p, axis=1)
    # Sphere radius
    R = np.max(normp)
    R = np.tile(R*(10**param), (numPts, 1))
    # Spherical flipping
    P=p+2*np.tile(R-normp.reshape([numPts, 1]),(1,dim)) *p /np.tile(normp.reshape([numPts, 1]),(1,dim))
    # convex hull
    aug_P = np.vstack((P, np.zeros((1, dim))))
    d = ConvexHull(aug_P)
    visiblePtInds= np.unique(d.vertices)
    inds = np.where(visiblePtInds==numPts)
    visiblePtInds = np.delete(visiblePtInds, inds)

    return visiblePtInds


def in_convex_polyhedron(convex_hull, points, vis=None):
    in_bool = np.zeros([points.shape[0]])
    #ori_set = np.asarray(convex_hull.points)
    ori_set = np.asarray(convex_hull[:,:2])
    d = ConvexHull(ori_set, False)
    ori_edge_index = np.sort(np.unique(d.vertices))
    # a = np.array([1,2,3])
    for i in range(points.shape[0]):
        new_set = np.row_stack([ori_set, points[i,:2]])
        d = ConvexHull(new_set)
        new_edge_index = np.sort(np.unique(d.vertices))
        if ori_edge_index.shape[0] == new_edge_index.shape[0]:
            in_bool[i] = (ori_edge_index==new_edge_index).all()
        else:
            in_bool[i] = False

    # sphere = o3d.geometry.TriangleMesh.create_sphere(1).translate(points[0])
    # if in_bool[i]:
    #     sphere.paint_uniform_color([0.2,0.2,1.0])
    # else:
    #     sphere.paint_uniform_color([1,0,0])

    # pcd_cur2 = o3d.geometry.PointCloud()
    # pcd_cur2.points = o3d.utility.Vector3dVector(convex_hull[ori_edge_index,:])
    # pcd_cur2.paint_uniform_color([0,0,1.0])
    # vis_list = [sphere, pcd_cur2]
    # for item in vis_list:
    #     vis.add_geometry(item)
    # vis.run()

    return bool(in_bool)


def readPointCloud(filename):
	"""
	reads bin file and returns
	as m*4 np array
	all points are in meters
	you can filter out points beyond(in x y plane)
	50m for ease of computation
	and above or below 10m
	"""
	pcl = np.fromfile(filename, dtype=np.float32,count=-1)
	pcl = pcl.reshape([-1,4])
	return pcl 


# def main():
#     # pc1 = np.load('./data/pc1.npy')
#     # pc2 = np.load('./data/pc2.npy')
#     pc1 = readPointCloud('./dataset/0000000789.bin')[:, :3]
#     pc2 = readPointCloud('./dataset/0000000790.bin')[:, :3]
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(pc1)
#     pcd.paint_uniform_color([0,1,0])
#     pcd2 = o3d.geometry.PointCloud()
#     pcd2.points = o3d.utility.Vector3dVector(pc2)
#     pcd2.paint_uniform_color([1,0,0])
#     vis_list = [pcd, pcd2]
#     # draw(None, vis_list, save_fov=False)
#     pcd3 = pcd + pcd2
#     diameter = np.linalg.norm(np.asarray(pcd3.get_max_bound()) - np.asarray(pcd3.get_min_bound()))
#     print("Define parameters used for hidden_point_removal")
#
#     camera = [-1.0, 0.0, 3]
#     mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3).translate(camera)
#     print("Get all points that are visible from given view point")
#     # radius = diameter * 10
#     # _, pt_map = pcd3.hidden_point_removal(camera, radius)
#     pc = np.vstack([pc1, pc2])
#     camera = [-2.0, -1.4, 3]
#
#     pt_map = HPR(pc, camera, 3)
#     print("Visualize result")
#     pcd3 = pcd3.select_by_index(pt_map)
#
#     # bool_in = in_convex_polyhedron(pcd3,np.array([camera]))
#     sphere = o3d.geometry.TriangleMesh.create_sphere(1.0).translate(camera)
#     bool_in = False
#     if bool_in:
#         sphere.paint_uniform_color([0.8,0.2,1.0])
#     else:
#         sphere.paint_uniform_color([1,0,0])
#     o3d.visualization.draw_geometries([pcd3, mesh, sphere])
#
#
#
#
# if __name__ == '__main__':
#     main()