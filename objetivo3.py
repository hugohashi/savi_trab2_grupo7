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
    #scene_number = random.choice(list(views.keys()))
    scene_number = '14'

    #Converter imagem em point cloud
    filename_rgb = f'images/{scene_number}-color.png'
    filename_depth = f'images/{scene_number}-depth.png'

    color_raw = o3d.io.read_image(filename_rgb)
    depth_raw = o3d.io.read_image(filename_depth)

    #criar imagem rgbd a partir da imagem rgb e respetiva depth
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_scale = 6000, convert_rgb_to_intensity = False)

    #matriz (w, h, fx, fy, cx, cy)
    K = o3d.camera.PinholeCameraIntrinsic(640, 480, 600, 600, 320, 240)    

    #Criar point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, K)

    #Transformation matrix
    #T = views[scene_number]['T']
    #print(T)

    # Criar a matriz de transformação (identidade 4x4)
    T = np.eye(4)

    #Adicionar Rotação
    T[:3, :3] = pcd.get_rotation_matrix_from_xyz((105 * math.pi /180 , 0*math.pi / 180 , 0*math.pi / 180))

    #Adicionar Translação
    T[0:3, 3] = [0, 0.27, 1.4]

    #Tranformar a pcd - para que o sistema de coordenadas esteja no centro da mesa
    pcd_downsampled = pcd.transform(np.linalg.inv(T))

    print(T)

    #Apply transformation
    #pcd_downsampled = pcd.transform(np.linalg.inv(T))

    #Create a vector3d with the points in the boundingbox
    #np_vertices = np.ndarray((8, 3), dtype=float)

    #sx = sy = 0.5
    #sz_top = 0.5
    #sz_bottom = -0.05

    #top vertices
    #np_vertices[0, 0:3] = [sx, sy, sz_top]
    #np_vertices[1, 0:3] = [sx, -sy, sz_top]
    #np_vertices[2, 0:3] = [-sx, -sy, sz_top]
    #np_vertices[3, 0:3] = [-sx, sy, sz_top]

    #bottom vertices
    #np_vertices[4, 0:3] = [sx, sy, sz_bottom]
    #np_vertices[5, 0:3] = [sx, -sy, sz_bottom]
    #np_vertices[6, 0:3] = [-sx, -sy, sz_bottom]
    #np_vertices[7, 0:3] = [-sx, sy, sz_bottom]

    #numpy to open3d
    #vertices = o3d.utility.Vector3dVector(np_vertices)

    #Create a bounding box
    #box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(vertices)

    #Crop the original point cloud using the bounding box
    #pcd_cropped = pcd_downsampled.crop(box)

    #remover plano mesa -> Plane segmentation - RANSAC
    #Para detetar os pontos que pertencem à mesa (1000 iterações, porque quantas mais, mais facil é identificar o chão, visto que é o maior plano, logo tem mais pontos!)
    #plane_model, inliers = pcd_cropped.segment_plane(distance_threshold = 0.01, ransac_n = 3, num_iterations = 100)

    #a, b, c, d = plane_model

    #nuvem só com os objetos em cima da mesa (outliers)
    #point_cloud_objects = pcd_cropped.select_by_index(inliers, invert = True)
    #point_cloud_table = pcd_cropped.select_by_index(inliers, invert = False)
    #point_cloud_table.paint_uniform_color([0, 1, 0])

    #Clustering - separar objetos!
    #labels = point_cloud_objects.cluster_dbscan(eps=0.02, min_points=100, print_progress=True)

    #groups = list(set(labels))

    #colormap = cm.Set1(range(0, len(groups)))
    #groups.remove(-1)

    #objects_point_clouds = []
    #caixas = []
    #i = 0

    #for group_n in groups:
        #encontrar os indices dos objetos que pertencem a um dado grupo!
        #group_idx = list(locate(labels, lambda x: x==group_n))

        #object_point_cloud = point_cloud_objects.select_by_index(group_idx, invert=False)
        
        #pintar de uma dada cor a caixa em volta do objeto encontrado
        #caixa = object_point_cloud.get_oriented_bounding_box()
        #caixa.color = colormap[group_n, 0:3]

        #guardar imagem do objeto - pontos que pertencem ao objeto são correspondentes aos pixeis da image
        #TO DO

        #caixas.append(caixa)      
        #objects_point_clouds.append(object_point_cloud)

    #----------------------
    # Visualization 
    #----------------------
    #criar sistema de coordenadas
    frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))

    entities = [pcd_downsampled]
    entities.append(frame)
    #entities = []
    #entities.extend([point_cloud_table])
    #entities.extend(caixas)
    o3d.visualization.draw_geometries(entities,
                                    zoom=views[scene_number]['trajectory'][0]['zoom'],
                                    front=views[scene_number]['trajectory'][0]['front'],
                                    lookat=views[scene_number]['trajectory'][0]['lookat'],
                                    up=views[scene_number]['trajectory'][0]['up'])


if __name__ == "__main__":
    main()