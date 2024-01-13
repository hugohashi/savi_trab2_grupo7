import open3d as o3d

pcd_full = o3d.io.read_point_cloud("/home/rita/Desktop/savi_trab2_grupo7/data/rgbd-scenes-v2/pc/01.ply")
#pcd_full.paint_uniform_color([1, 0.706, 0])

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd_full)
vis.update_geometry(pcd_full)
vis.poll_events()
vis.update_renderer()
vis.capture_screen_image("image1.jpg")
vis.destroy_window()