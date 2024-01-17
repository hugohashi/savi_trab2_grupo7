#!/usr/bin/env python3
import math
from copy import deepcopy
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
    

    def rgb_to_name(self,rgb):
            

            r, g, b = cv2.split(rgb)
            min_diff = float('inf')
            
            
            for name, hex_value in webcolors.CSS21_HEX_TO_NAMES
                #hex_rgb = tuple(int(hex_value[i:i+2], 16) for i in (1, 3, 5))
                #diff = sum(abs(a - b) for a, b in zip(hex_rgb, (r, g, b)))
                #if diff < min_diff:
                    #min_diff = diff
                    closest_name = name
            return closest_name

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
        

        ## try
        #if   r>80 and r<80 and b>100 and b< 80 and g>100 and g<80:
    
        lower_red =np.array([0,0,200], dtype = "uint8")
        upper_red =np.array([0,0,255], dtype = "uint8")

        lower_yellow =np.array([0,0,200], dtype = "uint8")
        upper_yellow =np.array([0,0,255], dtype = "uint8")

        lower_green =np.array([0,0,200], dtype = "uint8")
        upper_green =np.array([0,0,255], dtype = "uint8")

        lower_black =np.array([0,0,200], dtype = "uint8")
        upper_black =np.array([0,0,255], dtype = "uint8")

        lower_white =np.array([0,0,200], dtype = "uint8")
        upper_white =np.array([0,0,255], dtype = "uint8")

        lower_blue =np.array([0,0,200], dtype = "uint8")
        upper_blue =np.array([0,0,255], dtype = "uint8"
                             )
        mask = cv2.inRange(img, lower_red, upper_red)
        detected_output = cv2.bitwise_and(img, img, mask = mask)


        ##

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
        