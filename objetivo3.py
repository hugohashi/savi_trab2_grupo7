#CRIA NUVEM ATRAVÉS DUMA IMAGEM, ISOLA OS OBJETOS E DEVOLVE UMA IMAGEM DE CADA OBJETO ISOLADO

from views import *
import open3d as o3d
import cv2
import copy
import numpy as np
import math
import random
from matplotlib import cm
from more_itertools import locate
from classes import *

def main():
    #escolher a cena aleatoriamente
    scene_number = random.choice(list(views.keys()))

    #Converte imagem em point cloud
    filename_rgb = f'images/{scene_number}-color.png'
    filename_depth = f'images/{scene_number}-depth.png'

    image_rgb = cv2.imread(filename_rgb)
    color_raw = o3d.io.read_image(filename_rgb)
    depth_raw = o3d.io.read_image(filename_depth)

    #cria imagem rgbd a partir da imagem rgb e respetiva depth
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_scale = 6000, convert_rgb_to_intensity = False)

    #matriz intrinseca (w, h, fx, fy, cx, cy)
    K = o3d.camera.PinholeCameraIntrinsic(640, 480, 525, 525, 320, 240)    

    #Criar point cloud através da imagem RGBD e da matriz K
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, K)

    #matriz de transformação (de modo ao sistema de coordenadas ficar em cima da mesa)
    T = views[scene_number]['T']
    print(T)

    #Aplicar a matriz transformação à point cloud
    pcd_downsampled = pcd.transform(np.linalg.inv(T))

    #criar um vector3d com os pontos da caixa que envolverá a mesa e os objetos
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

    #numpy para open3d
    vertices = o3d.utility.Vector3dVector(np_vertices)

    #Criar a caixa
    box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(vertices)

    #Cortar a point cloud usando a caixa
    pcd_cropped = pcd_downsampled.crop(box)

    #remover plano mesa -> Plane segmentation - RANSAC
    #Para detetar os pontos que pertencem à mesa (100 iterações, porque quantas mais, mais facil é identificar a mesa, visto que é o maior plano e por isso tem mais pontos!)
    plane_model, inliers = pcd_cropped.segment_plane(distance_threshold = views[scene_number]['dist thr'], ransac_n = 3, num_iterations = 100)

    a, b, c, d = plane_model

    #nuvem só com os objetos em cima da mesa (outliers)
    point_cloud_objects = pcd_cropped.select_by_index(inliers, invert = True)
    #point cloud da mesa
    point_cloud_table = pcd_cropped.select_by_index(inliers, invert = False)
    #pintar mesa de verde
    point_cloud_table.paint_uniform_color([0, 1, 0])

    #Clustering -> separar objetos!
    labels = point_cloud_objects.cluster_dbscan(eps=views[scene_number]['eps'], min_points=200, print_progress=True)

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

        #desfazer transformação para converter as coordenadas 3D em 2D
        T_inv = np.linalg.inv(T)
        object_point_cloud = object_point_cloud.transform(np.linalg.inv(T_inv))

        #converter e encontrar os cantos da imagem (umax, umin, vmax, vmin)
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

        #criar sub imagem, recortando a imagem rgb inicial
        img = image_rgb[vmin:vmax, umin:umax]
        cv2.imwrite(f'object{i}_scene{scene_number}.png', img)

        #repor a transformação novamente
        object_point_cloud = object_point_cloud.transform(np.linalg.inv(T))

        caixas.append(caixa)
        objects_point_clouds.append(object_point_cloud)
        i = i + 1

    #----------------------
    # Visualização
    #----------------------
    
    #criar sistema de coordenadas
    #frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))

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
