#!/usr/bin/env python3

from copy import deepcopy
import open3d as o3d
import math
import numpy as np
import os
import cv2
from more_itertools import locate

import math
from copy import deepcopy
import open3d as o3d
import cv2
import numpy as np
import os
from gtts import gTTS
import pygame
from more_itertools import locate

class ObjectProperties():
    
    def __init__(self, object):
        self.idx = object['idx']
        self.center = object['center']

        self.point_cloud = object['points']
        pc_points = self.point_cloud.points
        self.points = np.asarray(pc_points)

    def getSize(self):
        self.point_cloud.translate(-self.center)
        pc_points_centered = self.point_cloud.points
        points = np.asarray(pc_points_centered)
        
        max_dist_from_center = 0
        max_z = -1000
        min_z = 1000
        for point in points:
            dist_from_center = math.sqrt(point[0]**2 + point[1]**2)

            if dist_from_center >= max_dist_from_center:
                max_dist_from_center = dist_from_center

            z = point[2]
            if z >= max_z:
                max_z = z
            elif z <= min_z:
                min_z = z
        
        width = max_dist_from_center*2
        height = abs(max_z - min_z)

        self.point_cloud.translate(self.center)

        return (width, height)

    def getColor(self, idx):  
        idx = idx + 1
        image_name = 'image' + str(idx) + '.png'

        # Creating o3d windows with only one object to then process in OpenCV
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(self.point_cloud)
        vis.get_view_control().rotate(0, np.pi / 4) # rotate around y-axis
        vis.get_view_control().set_zoom(3.0) #set the zoom level
        vis.run()  # user changes the view and press "q" to terminatem)
        vis.capture_screen_image(image_name)
        vis.destroy_window()

        # OpenCV processing
        img = cv2.imread(image_name)
        
        # print(str(img))
        colored_pixels = []

        b = 0
        g = 0
        r = 0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                pixel = img[i, j]

                if pixel[0] < 250 or pixel[1] < 250 or pixel[2] < 250:
                    colored_pixels.append(pixel)
                    b = b + pixel[0]
                    g = g + pixel[1]
                    r = r + pixel[2]

        b = b/len(colored_pixels)
        g = g/len(colored_pixels)
        r = r/len(colored_pixels)

        return (r,g,b)
    

class audioprocessing():
    def __init__(self):
        pass

    # --------------------------------------------------------------
    # 3D to pixel 
    # --------------------------------------------------------------
    def loadaudio(self, lista_audio, cenario, dimensions):
        
        cleaned_fruits = [fruit.split("_")[0] for fruit in lista_audio]
        # text = ""
        # for i, fruit in enumerate(cleaned_fruits):
        #     text += "The object number " + str(i+1) + " I think is a " + fruit + ". "        
        print(cleaned_fruits)
        print(lista_audio)
        print(len(dimensions))
        text = ""
        for i in range(len(dimensions)):
            dim = dimensions[i]
            fruta = cleaned_fruits[i]
            text += "The object number " + str(int(i + 1)) + ", a " + fruta + ", has dimensions of " + str(round(dim[0], 2)) + " x " + str(round(dim[1], 2)) + "."
        
        pygame.mixer.init()

        # Gerar a descrição da cena
        text_final = "We are looking ate the scene "+str(cenario)+" we have "+ str(len(lista_audio))+ " objects processed in the scene "+str(text)+" i think its everything if you have any questions you may ask to my creators, thanks professor i hope you like it"
        print(text_final)
        
        tts = gTTS(text_final, lang='en')
        tts.save("narracao.mp3")

        pygame.init()
        pygame.mixer.music.load("narracao.mp3")
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)    
        