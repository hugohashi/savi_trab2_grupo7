#isola os objetos e retira uma imagem de cada objeto

from views import *
import open3d as o3d
import cv2
import copy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import random
from matplotlib import cm
from more_itertools import locate
import webcolors
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import argparse
from classes import *


def main():
    #escolher cena aleatoriamente
    scene_number = random.choice(list(views.keys()))

    #Converter imagem em point cloud
    filename_rgb = f'images/{scene_number}-color.png'
    filename_depth = f'images/{scene_number}-depth.png'

    image_rgb = cv2.imread(filename_rgb)
    color_raw = o3d.io.read_image(filename_rgb)
    depth_raw = o3d.io.read_image(filename_depth)

    #criar imagem rgbd a partir da imagem rgb e respetiva depth
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_scale = 6000, convert_rgb_to_intensity = False)

    #matriz (w, h, fx, fy, cx, cy)
    K = o3d.camera.PinholeCameraIntrinsic(640, 480, 525, 525, 320, 240)    

    #Criar point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, K)

    T = views[scene_number]['T']
    print(T)

    #Apply transformation
    pcd_downsampled = pcd.transform(np.linalg.inv(T))

    #Create a vector3d with the points in the boundingbox
    np_vertices = np.ndarray((8, 3), dtype=float)

    sx = views[scene_number]['sx']
    sy = views[scene_number]['sy']
    sz_top = views[scene_number]['sz top']
    sz_bottom = views[scene_number]['sz bot']

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
    plane_model, inliers = pcd_cropped.segment_plane(distance_threshold = views[scene_number]['dist thr'], ransac_n = 3, num_iterations = 100)

    a, b, c, d = plane_model

    #nuvem só com os objetos em cima da mesa (outliers)
    point_cloud_objects = pcd_cropped.select_by_index(inliers, invert = True)
    #point cloud mesa pintada!
    point_cloud_table = pcd_cropped.select_by_index(inliers, invert = False)
    point_cloud_table.paint_uniform_color([0, 1, 0])

    #Clustering - separar objetos!
    labels = point_cloud_objects.cluster_dbscan(eps=views[scene_number]['eps'], min_points=200, print_progress=True)

    groups = list(set(labels))

    colormap = cm.Set1(range(0, len(groups)))
    groups.remove(-1)

    objects_point_clouds = []
    objects = []
    caixas = []
    i = 0

    for group_n in groups:
        #encontrar os indices dos objetos que pertencem a um dado grupo!
        group_idx = list(locate(labels, lambda x: x==group_n))

        object_point_cloud = point_cloud_objects.select_by_index(group_idx, invert=False)
        
        #pintar de uma dada cor a caixa em volta do objeto encontrado
        caixa = object_point_cloud.get_oriented_bounding_box()
        caixa.color = colormap[group_n, 0:3]

        #sub imagem
        #desfazer transformação
        T_inv = np.linalg.inv(T)
        object_point_cloud = object_point_cloud.transform(np.linalg.inv(T_inv))

        # Get the 3D coordinates of the points in the object_point_cloud
        umax = None
        umin = None
        vmax = None
        vmin = None
        for (x, y, z) in object_point_cloud.points:
            u = round(x*525/z + 320)
            v = round(y*525/z + 240)
            if (umax and umin) is None:
                umax = u
                umin = u
            elif u > umax:
                umax = u
            elif u < umin:
                umin = u
            if (vmax and vmin) is None:
                vmax = v
                vmin = v
            elif v > vmax:
                vmax = v
            elif v < vmin:
                vmin = v       

        img = image_rgb[vmin-20:vmax+20, umin-20:umax+20]
        cv2.imwrite(f'objetos/object{i}.png', img)

        #repor transformação
        object_point_cloud = object_point_cloud.transform(np.linalg.inv(T))

        # Create a dictionary to represent the objects
        d = {}
        d['idx'] = str(group_n + 1)
        d['points'] = object_point_cloud
        d['center'] = d['points'].get_center()
        
        objects.append(d)

        pc_to_convert = d["points"]
        pc_points = pc_to_convert.points
        points = np.asarray(pc_points)

        caixas.append(caixa)
        objects_point_clouds.append(object_point_cloud)
        i = i + 1

    #cor e volume
    color = []
    dimensions = []
    i = 0
    for group_n, object in enumerate(objects):

        properties = ObjectProperties(object)
        size = properties.getSize()
        volume = math.pi* size[0]**2 * size[1] #cylindrical volume, aproximation
        print(" \n" f"The object {i} volume is " + str(volume))
        
        dimensions.append(size)
        color_rgb = properties.getColor(group_n)
        color_rgb_try=properties.getColortry(group_n)

        
        rgb1,rgb2,rgb3 = properties.getcolortry2(group_n)
        ttt= [rgb1,rgb2,rgb3]
        min_colours = {}
        
        
        closest_names =[]
        for p in ttt :
            closest_name = None
            for key, name in webcolors.CSS21_HEX_TO_NAMES.items():#CSS3_HEX_TO_NAMES.items():
                #r, g, b = cv2.split(color_rgb_try)
                
               # r, g, b = cv2.split(p)
                r, g, b = p
                r, g, b = webcolors.hex_to_rgb(key)
                rd = (r - color_rgb[0]) ** 2
                gd = (g - color_rgb[1]) ** 2
                bd = (b - color_rgb[2]) ** 2
                min_colours[(rd + gd + bd)] = name
            closest_color = min_colours[min(min_colours.keys())]        

            try:
                actual_name = webcolors.rgb_to_name(color_rgb)
                closest_name = actual_name
            except ValueError:
                closest_name = closest_color
                actual_name = None
            closest_names.append(closest_name)
        
        
        print(f"The object {i} approximate colors is " + str(closest_names) + ' with ' + str(color_rgb_try) + ' RGB value')
        
        color.append(closest_name)

        i = i + 1


    #visualização

    entities = [pcd_cropped]
    entities.extend([point_cloud_table])
    entities.extend(objects_point_clouds)
    entities.extend(caixas)
    o3d.visualization.draw_geometries(entities,
                                    zoom=views[scene_number]['trajectory'][0]['zoom'],
                                    front=views[scene_number]['trajectory'][0]['front'],
                                    lookat=views[scene_number]['trajectory'][0]['lookat'],
                                    up=views[scene_number]['trajectory'][0]['up'])

if __name__ == "__main__":
    main()