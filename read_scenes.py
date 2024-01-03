
#!/usr/bin/env python3

from views import *

import open3d as o3d
import numpy as np
import math
import random

def main():
    scene_number = random.choice(list(views.keys()))

    pcd_original = o3d.io.read_point_cloud(f"data/scenes/rgbd-scenes-v2_pc/rgbd-scenes-v2/pc/{scene_number}.ply")

    # Downsample using voxel grid ------------------------------------
    pcd_downsampled = pcd_original.voxel_down_sample(voxel_size=0.02)
    # pcd_downsampled.paint_uniform_color([1,0,0])

    # estimate normals
    pcd_downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd_downsampled.orient_normals_to_align_with_direction(orientation_reference=np.array([0, 0, 1]))

    # Create transformation T1 only with rotation
    T1 = np.zeros((4, 4), dtype=float)

    # Add homogeneous coordinates
    T1[3, 3] = 1

    # Add null rotation
    R = pcd_downsampled.get_rotation_matrix_from_xyz((110*math.pi/180, 0, 40*math.pi/180))
    T1[0:3, 0:3] = R
    # T[0:3, 0] = [1, 0, 0]  # add n vector
    # T[0:3, 1] = [0, 1, 0]  # add s vector
    # T[0:3, 2] = [0, 0, 1]  # add a vector

    # Add a translation
    T1[0:3, 3] = [0, 0, 0]
    print('T1=\n' + str(T1))

    # Create transformation T2 only with translation
    T2 = np.zeros((4, 4), dtype=float)

    # Add homogeneous coordinates
    T2[3, 3] = 1

    # Add null rotation
    T2[0:3, 0] = [1, 0, 0]  # add n vector
    T2[0:3, 1] = [0, 1, 0]  # add s vector
    T2[0:3, 2] = [0, 0, 1]  # add a vector

    # Add a translation
    T2[0:3, 3] = [0.8, 1, -0.4]
    print('T2=\n' + str(T2))

    T = np.dot(T1, T2)
    print('T=\n' + str(T))

    # Create table ref system and apply transformation to it
    frame_table = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2, origin=np.array([0., 0., 0.]))

    frame_table = frame_table.transform(T)

    pcd_downsampled = pcd_downsampled.transform(np.linalg.inv(T))

    # Ex3 - Create crop the points in the table

    # Create a vector3d with the points in the boundingbox
    np_vertices = np.ndarray((8, 3), dtype=float)

    sx = sy = 0.6
    sz_top = 0.4
    sz_bottom = -0.1
    np_vertices[0, 0:3] = [sx, sy, sz_top]
    np_vertices[1, 0:3] = [sx, -sy, sz_top]
    np_vertices[2, 0:3] = [-sx, -sy, sz_top]
    np_vertices[3, 0:3] = [-sx, sy, sz_top]
    np_vertices[4, 0:3] = [sx, sy, sz_bottom]
    np_vertices[5, 0:3] = [sx, -sy, sz_bottom]
    np_vertices[6, 0:3] = [-sx, -sy, sz_bottom]
    np_vertices[7, 0:3] = [-sx, sy, sz_bottom]

    print('np_vertices =\n' + str(np_vertices))

    vertices = o3d.utility.Vector3dVector(np_vertices)

    # Create a bounding box
    bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(vertices)
    print(bbox)

    # Crop the original point cloud using the bounding box
    pcd_cropped = pcd_downsampled.crop(bbox)

    # Visualization ----------------------
    pcd_downsampled.paint_uniform_color([0.4, 0.3, 0.3])
    pcd_cropped.paint_uniform_color([0.9, 0.0, 0.0])
    pcds_to_draw = [pcd_downsampled, pcd_cropped]

    frame_world = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))

    entities = []
    entities.append(frame_world)
    # entities.append(frame_table)
    entities.extend(pcds_to_draw)
    o3d.visualization.draw_geometries(entities,
                                      zoom=0.3412,
                                      front=views[scene_number]['trajectory'][0]['front'],
                                      lookat=views[scene_number]['trajectory'][0]['lookat'],
                                      up=views[scene_number]['trajectory'][0]['up'], point_show_normal=False)


if __name__ == "__main__":
    main()
