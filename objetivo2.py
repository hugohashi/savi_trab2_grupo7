#2.1 - numa cena isolar objetos

from views import *
import open3d as o3d
import copy
import numpy as np
import math
import random
from matplotlib import cm
from more_itertools import locate

def main():
    #escolher cena aleatoriamente
    scene_number = random.choice(list(views.keys()))

    #f"data/scenes/rgbd-scenes-v2_pc/rgbd-scenes-v2/pc/{scene_number}.ply")
    pcd_original = o3d.io.read_point_cloud(f'/home/rita/Desktop/savi_trab2_grupo7/data/rgbd-scenes-v2/pc/{scene_number}.ply')

    pcd_downsampled = copy.deepcopy(pcd_original)
    #Downsample using voxel grid ------------------------------------
    #pcd_downsampled = pcd_original.voxel_down_sample(voxel_size=0.02)
    #pcd_downsampled.paint_uniform_color([1,0,0])

    #Transformation matrix
    T = views[scene_number]['T']
    print(T)

    #Apply transformation
    pcd_downsampled = pcd_downsampled.transform(np.linalg.inv(T))

    #Create a vector3d with the points in the boundingbox
    np_vertices = np.ndarray((8, 3), dtype=float)

    sx = sy = 0.5
    sz_top = 0.5
    sz_bottom = -0.05

    #top vertices
    np_vertices[0, 0:3] = [sx, sy, sz_top]
    np_vertices[1, 0:3] = [sx, -sy, sz_top]
    np_vertices[2, 0:3] = [-sx, -sy, sz_top]
    np_vertices[3, 0:3] = [-sx, sy, sz_top]

    #bottom vertices
    np_vertices[4, 0:3] = [sx, sy, sz_bottom]
    np_vertices[5, 0:3] = [sx, -sy, sz_bottom]
    np_vertices[6, 0:3] = [-sx, -sy, sz_bottom]
    np_vertices[7, 0:3] = [-sx, sy, sz_bottom]

    #numpy to open3d
    vertices = o3d.utility.Vector3dVector(np_vertices)

    #Create a bounding box
    box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(vertices)

    #Crop the original point cloud using the bounding box
    pcd_cropped = pcd_downsampled.crop(box)

    #remover plano mesa -> Plane segmentation - RANSAC
    #Para detetar os pontos que pertencem à mesa (1000 iterações, porque quantas mais, mais facil é identificar o chão, visto que é o maior plano, logo tem mais pontos!)
    plane_model, inliers = pcd_cropped.segment_plane(distance_threshold = 0.01, ransac_n = 3, num_iterations = 100)

    a, b, c, d = plane_model

    #nuvem só com os objetos em cima da mesa (outliers)
    point_cloud_objects = pcd_cropped.select_by_index(inliers, invert = True)

    #Clustering - separar objetos!
    labels = point_cloud_objects.cluster_dbscan(eps=0.02, min_points=100, print_progress=True)

    groups = list(set(labels))

    colormap = cm.Set1(range(0, len(groups)))
    groups.remove(-1)

    objects_point_clouds = []
    caixas = []
    i = 0

    for group_n in groups:
        #encontrar os indices dos objetos que pertencem a um dado grupo!
        group_idx = list(locate(labels, lambda x: x==group_n))

        object_point_cloud = point_cloud_objects.select_by_index(group_idx, invert=False)
        
        #pintar de uma dada cor a caixa em volta do objeto encontrado
        caixa = object_point_cloud.get_oriented_bounding_box()
        caixa.color = colormap[group_n, 0:3]

        #guardar imagem do objeto
        #<scene>.pose - where each line is the camera pose (a,b,c,d,x,y,z) for one video frame. 
        #a,b,c,d is the camera orientation expressed as a quarternion, 
        #and x,y,z is the camera position in the global coordinate frame. 
        #The camera pose of the first frame of each video is always the origin of the global coordinate frame.

        caixas.append(caixa)      
        objects_point_clouds.append(object_point_cloud)

    #----------------------
    # Visualization 
    #----------------------
    #criar sistema de coordenadas
    frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))
    #pcd_cropped.paint_uniform_color([0.3, 0.3, 0.3])
    #entities = [pcd_cropped]
    entities = []
    entities.append(frame)
    entities.extend(objects_point_clouds)
    entities.extend(caixas)
    o3d.visualization.draw_geometries(entities,
                                    zoom=0.3412,
                                    front=views[scene_number]['trajectory'][0]['front'],
                                    lookat=views[scene_number]['trajectory'][0]['lookat'],
                                    up=views[scene_number]['trajectory'][0]['up'], point_show_normal=False)


if __name__ == "__main__":
    main()