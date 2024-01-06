from copy import deepcopy
import math
import open3d as o3d
import numpy as np
from matplotlib import cm
from more_itertools import locate

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


# -----------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------
pcd_downsampled = o3d.io.read_point_cloud('/home/rita/Desktop/savi_trab2_grupo7/data/scenes/rgbd-scenes-v2/pc/01.ply')

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

colormap = cm.Set1(range(0, len(groups)))
groups.remove(-1)

objects_point_clouds = []

for group_n in groups:
    #encontrar os indices dos objetos que pertencem a um dado grupo!
    group_idx = list(locate(labels, lambda x: x==group_n))

    object_point_cloud = point_cloud_objects.select_by_index(group_idx, invert=False)
    
    #pintar de uma dada cor o grupo encontrado
    color = colormap[group_n, 0:3]
    object_point_cloud.paint_uniform_color(color)
    objects_point_clouds.append(object_point_cloud)

#----------------------
# Visualization 
#----------------------
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
