from copy import deepcopy
import math
import open3d as o3d
import numpy as np
from matplotlib import cm
from more_itertools import locate
import webcolors
from classes import *


view = {
    "class_name": "ViewTrajectory",
    "interval": 29,
    "is_loop": False,
    "trajectory":
        [
            {
                "boundingbox_max": [2.6540005122611348, 2.3321821423160629, 0.85104994623420782],
                "boundingbox_min": [-2.5261458770339673, -2.1656718060235378, -0.55877501755379944],
                "field_of_view": 60.0,
                "front": [0.75672239933786944, 0.34169632162348007, 0.55732830013316348],
                "lookat": [0.046395260625899069, 0.011783639768603466, -0.10144691776517496],
                "up": [-0.50476400916821107, -0.2363660920597864, 0.83026764695055955],
                "zoom": 0.30119999999999997
            }
        ],
    "version_major": 1,
    "version_minor": 0
}

def main():
    # -----------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------
    pcd_downsampled = o3d.io.read_point_cloud('data/scenes/rgbd-scenes-v2/pc/01.ply')

    # -----------------------------------------------------------------
    # Execution
    # -----------------------------------------------------------------

    # Matriz transformação
    T = [[0.766, -0.643, 0, -0.03], [-0.22, -0.262, -0.94, -0.156], [0.604, 0.72, -0.342, 1.306], [0, 0, 0, 1]]

    # Tranformar a pcd 
    pcd_downsampled = pcd_downsampled.transform(np.linalg.inv(T))

    # Criar vértices da caixa - numpy
    np_vertices = np.ndarray((8, 3), dtype=float)

    sx = sy = 0.5
    sz_top = 0.6
    sz_bottom = -0.05

    #vértices do topo
    np_vertices[0, 0:3] = [sx, sy, sz_top]
    np_vertices[1, 0:3] = [sx, -sy, sz_top]
    np_vertices[2, 0:3] = [-sx, -sy, sz_top]
    np_vertices[3, 0:3] = [-sx, sy, sz_top]

    #vértices da base
    np_vertices[4, 0:3] = [sx, sy, sz_bottom]
    np_vertices[5, 0:3] = [sx, -sy, sz_bottom]
    np_vertices[6, 0:3] = [-sx, -sy, sz_bottom]
    np_vertices[7, 0:3] = [-sx, sy, sz_bottom]

    #Passar os vértices criados para open3d
    vertices = o3d.utility.Vector3dVector(np_vertices)

    #Criar a caixa através dos vértices definidos anteriormente
    box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(vertices)
    box.color = (1,0,0)

    # cortar a pcd para conter apenas os pontos dentro da caixa
    pcd_cropped = pcd_downsampled.crop(box)

    #remover plano mesa -> Plane segmentation - RANSAC
    #Para detetar os pontos que pertencem à mesa (1000 iterações, porque quantas mais, mais facil é identificar o chão, visto que é o maior plano, logo tem mais pontos!)
    plane_model, inliers = pcd_cropped.segment_plane(distance_threshold = 0.01, ransac_n = 3, num_iterations = 100)

    a, b, c, d = plane_model

    #nuvem só com os objetos em cima da mesa (outliers)
    point_cloud_objects = pcd_cropped.select_by_index(inliers, invert = True)

    #Clustering - separar objetos!
    labels = point_cloud_objects.cluster_dbscan(eps=0.02, min_points=500, print_progress=True)

    groups = list(set(labels))

    #colormap = cm.Set1(range(0, len(groups)))
    #groups.remove(-1)

    objects_point_clouds = []

    for group_n in groups:
        #encontrar os indices dos objetos que pertencem a um dado grupo!
        group_idx = list(locate(labels, lambda x: x==group_n))

        object_point_cloud = point_cloud_objects.select_by_index(group_idx, invert=False)
        
        #pintar de uma dada cor o grupo encontrado
        #color = colormap[group_n, 0:3]
        #object_point_cloud.paint_uniform_color(color)
        objects_point_clouds.append(object_point_cloud)
    
    frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.3, origin=np.array([0., 0., 0.]))

    outliers = point_cloud_objects
    print(outliers)
    
    cluster_idxs = list(outliers.cluster_dbscan(eps=0.03, min_points=10, print_progress=True))
    object_idxs = list(set(cluster_idxs))
    object_idxs.remove(-1)

    objects = []
    for object_idx in object_idxs:

        object_point_idxs = list(locate(cluster_idxs, lambda x: x == object_idx))
        object_points = outliers.select_by_index(object_point_idxs)
        # Create a dictionary to represent the objects
        d = {}
        d['idx'] = str(object_idx + 1)
        d['points'] = object_points 
        d['center'] = d['points'].get_center()
        

        pc_to_convert = d["points"]
        pc_points = pc_to_convert.points
        points = np.asarray(pc_points)
        
        if points.size > 700:
            objects.append(d) # add the dict of this object to the list
        else:
            continue
    
    #----------------------
    # Visualization 
    #----------------------
        
        
    entities = []
    
    entities.append(frame)
    
    # Draw bbox
    bbox_to_draw = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(box)
    entities.append(bbox_to_draw)
    dimensions = []
    # Draw objects
    color = []
    for object_idx, object in enumerate(objects):
        entities.append(object['points'])

        properties = ObjectProperties(object)
        size = properties.getSize()
        print("This object's volume is " + str(size))
        dimensions.append(size)

        color_rgb = properties.getColor(object_idx)
        
        min_colours = {}
        for key, name in webcolors.CSS21_HEX_TO_NAMES.items():#CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - color_rgb[0]) ** 2
            gd = (g_c - color_rgb[1]) ** 2
            bd = (b_c - color_rgb[2]) ** 2
            min_colours[(rd + gd + bd)] = name
        closest_color = min_colours[min(min_colours.keys())]

        try:
            actual_name = webcolors.rgb_to_name(color_rgb)
            closest_name = actual_name
        except ValueError:
            closest_name = closest_color
            actual_name = None

        print("This object's approximate COLOR is " + str(closest_name) + ' with ' + 
              str(color_rgb) + ' RGB value')
        color.append(closest_name)
        # Get the aligned bounding box of the point cloud
        bbox_to_draw_object_processed = o3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(object['points'])
        entities.append(bbox_to_draw_object_processed)

    entities.append(pcd_downsampled)   
 
    #criar sistema de coordenadas no sitio correto, apos mover a point_cloud
    frame_table = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))
    entities = []
    entities.append(frame_table)
    entities.extend(objects_point_clouds)
    o3d.visualization.draw_geometries(entities,
                                        zoom=0.3412,
                                        front=view['trajectory'][0]['front'],
                                        lookat=view['trajectory'][0]['lookat'],
                                        up=view['trajectory'][0]['up'])
    
    
    

if __name__ == "__main__":
    main()
