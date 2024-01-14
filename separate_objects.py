from copy import deepcopy
import math
import open3d as o3d
import numpy as np
from matplotlib import cm
from more_itertools import locate
import webcolors
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import argparse
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



    frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.3, origin=np.array([0., 0., 0.]))

    outliers = point_cloud_objects
    print(outliers)
    
    cluster_idxs = list(outliers.cluster_dbscan(eps=0.03, min_points=200, print_progress=True))
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

    


    path = 'Data_objects'
    files = [f for f in os.listdir(path) if f.endswith('.pcd')]

    objects_point_clouds = []
    list_pcd = {}
    for group_n in groups:
        #encontrar os indices dos objetos que pertencem a um dado grupo!
        group_idx = list(locate(labels, lambda x: x==group_n))

        object_point_cloud = point_cloud_objects.select_by_index(group_idx, invert=False)
        
        #pintar de uma dada cor o grupo encontrado
        #color = colormap[group_n, 0:3]
        #object_point_cloud.paint_uniform_color(color)
        objects_point_clouds.append(object_point_cloud)
        #list_pcd[group_n]= {'object_point_cloud': object_point_cloud,'indexed': group_n}
    
    for i, file in enumerate(files):
        variable_name = os.path.splitext(file)[0]
        point_cloud = o3d.io.read_point_cloud(os.path.join(path, file))
        list_pcd[variable_name] = {'point_cloud': point_cloud, 'indexed': i} 
    
    list_pcd_model = []
    for variable_name, info in list_pcd.items():
        list_pcd_model.append(info["point_cloud"])
        
    
    for object_idx, object in enumerate(objects):
            object['rmse'] = 10
            object['indexed'] = 100
            min_error = 0.03
            for model_idx, models_object in enumerate(list_pcd_model): 
                #print("Apply point-to-point ICP to object " + str(object['idx']) )

                trans_init = np.asarray([[1, 0, 0, 0],
                                        [0,1,0,0],
                                        [0,0,1,0], 
                                        [0.0, 0.0, 0.0, 1.0]])
                reg_p2p = o3d.pipelines.registration.registration_icp(object['points'], models_object, 1.0, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
                
                
                # -----------------------------------------------------
                # Start processing each object and respectiv properties
                # -----------------------------------------------------
                ##Bounding box to see better the comparation###
                bbox_to_draw_object = o3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(object['points'])
                bbox_to_draw_object.color = (1, 0, 0)
                bbox_to_draw_object_target = o3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(models_object)
                bbox_to_draw_object_target.color = (0, 1, 0)
            
                ##Get some information about the bound boxes###
                Volume_source = o3d.geometry.AxisAlignedBoundingBox.volume(bbox_to_draw_object)
                Volume_target = o3d.geometry.AxisAlignedBoundingBox.volume(bbox_to_draw_object_target)
                
                # ------------------------------------------
                # Doing Some match for better analysis
                # ------------------------------------------
                volume_compare = abs(Volume_source - Volume_target)  
           
                # ------------------------------------------
                # Start associating each object 
                # ------------------------------------------
                
                if  volume_compare < 0.006 :        
                    if reg_p2p.inlier_rmse < min_error and reg_p2p.inlier_rmse != 0:
                        if object['rmse'] > reg_p2p.inlier_rmse:
                            object['rmse'] = reg_p2p.inlier_rmse
                            object['indexed'] = model_idx
                            object["fitness"] = reg_p2p.fitness
                            
                            

 
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
        volume = math.pi* size[0]**2 * size[1] #cylindrical volume
        #forlmula não está correta, aporximiação grosseira
        
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


        ## tentar descobrir a moda da cor dos pontos



        try:
            actual_name = webcolors.rgb_to_name(color_rgb)
            closest_name = actual_name
        except ValueError:
            closest_name = closest_color
            actual_name = None
        print(" \n" "This object's volume is " + str(volume))
        print("This object's approximate color is " + str(closest_name) + ' with ' + str(color_rgb) + ' RGB value')
        color.append(closest_name)
        # Get the aligned bounding box of the point cloud
        bbox_to_draw_object_processed = o3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(object['points'])
        entities.append(bbox_to_draw_object_processed)

    entities.append(pcd_downsampled)   
 
    #criar sistema de coordenadas no sitio correto, apos mover a point_cloud
    frame_table = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))
   # entities = []
    entities.append(frame_table)
    entities.extend(objects_point_clouds)
    o3d.visualization.draw_geometries(entities,
                                        zoom=0.3412,
                                        front=view['trajectory'][0]['front'],
                                        lookat=view['trajectory'][0]['lookat'],
                                        up=view['trajectory'][0]['up'])



    
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--dataset_path", help="Select the path of the desire point cloud", default="Matias/Scenes/pc/03.ply", type=str)
    args = parser.parse_args()

    
    #file_name = args.dataset_path.split('/')[-1]
    file_name = args.dataset_path.split('/')[-1]
    number = file_name.split('.')[0]
    scenario_name = str(number) 
    #image = ImageProcessing()
    #result = image.loadPointCloud(centers, args.cropped, number)
    lista_audio = []

 
    app = gui.Application.instance
    app.initialize() # create an open3d app

    w = app.create_window("Open3D - 3D Text", 1920, 1080)
    widget3d = gui.SceneWidget()
    widget3d.scene = rendering.Open3DScene(w.renderer)
    widget3d.scene.set_background([0,0,0,1])  # set black background
    material = rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.point_size = 2 * w.scaling



    for entity_idx, entity in enumerate(entities):
        widget3d.scene.add_geometry("Entity " + str(entity_idx), entity, material)
    # Draw labels
    for object_idx, object in enumerate(objects):
        label_pos = [object['center'][0], object['center'][1], object['center'][2] + 0.15]
        label_text = "Obj: " + object['idx']
        for i, object_point_cloud in enumerate(list_pcd.values()):
            print("objecto"+str(object["indexed"]))
            print("o i: "+str(i))
            if object['indexed'] == i:
                variable_name = list(list_pcd.keys())[i]
                print("nome da variável:", variable_name)
                label_text += " "+variable_name
                lista_audio.append(variable_name)
        label = widget3d.add_3d_label(label_pos, label_text)
        label.color = gui.Color(255,0,0)
        label.scale = 2
    bbox = widget3d.scene.bounding_box
    widget3d.setup_camera(60.0, bbox, bbox.get_center())
    w.add_child(widget3d)
    app.run()
    
    print(dimensions)
    print('.')
    print(lista_audio)
     # Inicialize audio processing
    audio = audioprocessing()
    audio_final = audio.loadaudio(lista_audio, number, dimensions)

if __name__ == "__main__":
    main()
