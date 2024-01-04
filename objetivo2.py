#2.1 - numa cena isolar objetos
#2.2 - calcular as suas propriedades

from views import *

import open3d as o3d
import numpy as np
import math
import random

def main():
  
  scene_number = random.choice(list(views.keys()))

  pcd_original = o3d.io.read_point_cloud(f"data/scenes/rgbd-scenes-v2_pc/rgbd-scenes-v2/pc/{scene_number}.ply")

  # Downsample using voxel grid
  pcd_downsampled = pcd_original.voxel_down_sample(voxel_size=0.02)

  # Create coordinates systems 
  frame = o3d.geometry.TriangleMesh().create_coordinate_frame()

  #PROBLEMA CADA CENA TERÁ A SUA MATRIZ DE TRANFORMAÇÃO E A SUA PRÓPRIA CAIXA, TENHO DE ALTERAR ISTO DE MODO TER ESTES PARAMETROS PARA CADA CENA AUTOMATICAMENTE!
  #Transformation matrix 4x4 
  T = np.eye(4)

  # Add homogeneous coordinates
  T[3, 3] = 1

  # Rotation - falta confirmar estes parametros
  T[:3, :3] = pcd_downsampled.get_rotation_matrix_from_xyz((110*math.pi/180, 0, 40*math.pi/180))

  #Translation [tx, ty, tz]
  T[0:3, 3] = [0.8, 1, -0.4]

  # Apply transformation to the coordinate system
  frame_table = frame_table.transform(T)

  #para que serve isto???
  #pcd_downsampled = pcd_downsampled.transform(np.linalg.inv(T))

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

  vertices = o3d.utility.Vector3dVector(np_vertices)

  # Create a bounding box
  box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(vertices)
  # Crop the original point cloud using the bounding box
  pcd_cropped = pcd_downsampled.crop(bbox)

  #Paint
  pcd_downsampled.paint_uniform_color([0.4, 0.3, 0.3])
  pcd_cropped.paint_uniform_color([1.0, 0.0, 0.0])
  pcds_to_draw = [pcd_downsampled, pcd_cropped]

  #visualization
  entities = []
  entities.append(frame_table)
  entities.extend(pcds_to_draw)
  o3d.visualization.draw_geometries(entities,
                                    zoom=0.3412,
                                    front=views[scene_number]['trajectory'][0]['front'],
                                    lookat=views[scene_number]['trajectory'][0]['lookat'],
                                    up=views[scene_number]['trajectory'][0]['up'])

if __name__ == "__main__":
    main()
