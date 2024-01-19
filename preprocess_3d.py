
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

        img = image_rgb[vmin:vmax, umin:umax]
        cv2.imwrite(f'object{i}_scene{scene_number}.png', img)

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