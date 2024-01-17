#!/usr/bin/env python3
import math
from copy import deepcopy
from uuid import RFC_4122
import open3d as o3d
import cv2
import numpy as np
import os
from gtts import gTTS
import pygame
from scipy.spatial import distance
import webcolors
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
    
    def getcolortry2(self,idx):
        idx = idx 
        path = 'objetos/object' + str(idx) + '.png'
        img = cv2.imread(path)

        data = img.reshape((-1, 3))
        data = np.float32(data)

        number_clusters = 5
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, centers = cv2.kmeans(data, number_clusters, None, criteria, 10, flags)

        r1 = centers[0][2]
        r2 = centers[1][2]
        r3 = centers[2][2]

        g1 = centers[0][1]
        g2 = centers[1][1]
        g3 = centers[2][1]

        b1 = centers[0][0]
        b2 = centers[1][0]
        b3 = centers[2][0]

        rgb1 = r1,g1,b1
        rgb2 = r2,g2,b2
        rgb3 = r3,g3,b3

        
        return (rgb1, rgb2, rgb3)
        

    def getColortry(self, idx): 
        # Image idx
        idx = idx 
        image_name = 'objetos/object' + str(idx) + '.png'

        # Load the PNG image
        image = cv2.imread(image_name)

        # Convert the image to the Lab color space

        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


        # Reshape the image into a list of pixels
        pixels = image_lab.reshape((-1, 3))

        # Calculate the mean 'L', 'a', and 'b' values
        mean_color = np.mean(pixels, axis=0)

        # Calculate the color distance between each pixel and the mean color

        distances = [distance.euclidean(pixel, mean_color) for pixel in pixels]

        # Find the pixel with the smallest color distance

        dominant_pixel = pixels[np.argmin(distances)]

        # Convert the dominant pixel to RGB for display

        dominant_color_rgb = cv2.cvtColor(dominant_pixel.reshape((1, 1, 3)), cv2.COLOR_LAB2BGR)
        dominant_color_rgb = cv2.cvtColor(dominant_color_rgb, cv2.COLOR_BGR2RGB)
        

        return (dominant_color_rgb)
    

    def getColor(self, idx):  
        idx = idx 
        image_name = 'objetos/object' + str(idx) + '.png'

        # OpenCV processing
        img = cv2.imread(image_name)
        
        # print(str(img))
        colored_pixels = []

        b = 0
        g = 0
        r = 0
        
        for i in range(img.shape[0]): # height
            for j in range(img.shape[1]): #width
                pixel = img[i, j]
                
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
       
        if len(cleaned_fruits) != len(dimensions):
            print("Error: Length mismatch between cleaned_fruits and dimensions")
            print("cleaned_fruits length:", len(cleaned_fruits))
            print("dimensions length:", len(dimensions))
            return
        
        print(cleaned_fruits)
        print(lista_audio)
        print(len(dimensions))
        text = ""
        for i in range(len(dimensions)):
            dim = dimensions[i]
            fruta = cleaned_fruits[i]
            text += "the object number " + str(int(i + 1)) + ", a " + fruta + ", has dimensions of " + str(round(dim[0], 2)) + " x " + str(round(dim[1], 2)) + "."
        
        pygame.mixer.init()

        # Gerar a descrição da cena
        text_final = "We are looking at the scene "+str(cenario)+" and I recognized "+ str(len(lista_audio))+ " objects processed in the scene, starting with "+str(text)+ "thank you for listening, hope I did not miss anything."
        print(text_final)
        
        tts = gTTS(text_final, lang='en')
        tts.save("narracao.mp3")

        pygame.init()
        pygame.mixer.music.load("narracao.mp3")
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)    
        