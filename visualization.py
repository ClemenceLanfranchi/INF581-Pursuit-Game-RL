#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 18:19:26 2020

@author: ireneburger
"""

from PIL import Image, ImageDraw
import copy
k=30
size = 30
champ = 3

class ImageResult :
    
    def __init__(self,k,size,champ):
        self.k=k
        self.size = size
        self.champ=champ
        self.image = Image.new('RGBA',[(k+2)*size,(k+2)*size],color =  "white")

    def draw_grid(self):
        k = (self.k)
        size = self.size
        draw = ImageDraw.Draw(self.image)
        for i in range (1,k+2):
            draw.line([(i*size,(k+1)*size),(i*size,size)],fill = (0,0,0,255),width = 2)
            draw.line([((k+1)*size,i*size),(size,i*size)],fill = (0,0,0,255),width = 2)
        draw.line([((k+1)*size,size),(size,size)],fill = (0,0,0,255),width =5)
        draw.line([(size,(k+1)*size),(size,size)],fill = (0,0,0,255),width =5)
        draw.line([(size,(k+1)*size),((k+1)*size,(k+1)*size)],fill = (0,0,0,255),width =5)
        draw.line([((k+1)*size,size),((k+1)*size,(k+1)*size)],fill = (0,0,0,255),width =5)

    
    def draw_case(self,xy,prey):
        if prey :
            color = (200,0,0,255)
        else :
            color = (0,200,0,255)
        size = self.size
        draw = ImageDraw.Draw(self.image)
        draw.ellipse([((xy[0]+1)*size,(xy[1]+1)*size),((xy[0]+2)*size,(xy[1]+2)*size)],fill = color)
    
    def draw_shade(self,xy):
        k=self.k
        size = self.size
        champ = self.champ
        draw = ImageDraw.Draw(self.image)
        poly = [(max(size,(xy[0]+1-champ)*size),max(size,(xy[1]+1-champ)*size)),(max(size,(xy[0]+1-champ)*size),min(size*(k+1),(xy[1]+2+champ)*size)),(min(size*(k+1),(xy[0]+2+champ)*size),min(size*(k+1),(xy[1]+2+champ)*size)),(min(size*(k+1),(xy[0]+2+champ)*size),max(size,(xy[1]+1-champ)*size))]
        draw.polygon(poly,fill = (200,255,200,255))

    
    def show(self):
        self.image.show()
        
    def draw_obs(self,positions):
        for i in range(1,len(positions)):
            self.draw_shade(list(positions[i]))
        for i in range(1,len(positions)):
            self.draw_case(list(positions[i]),False)
        self.draw_case(list(positions[0]),True)
        self.draw_grid()
        return self.image

        
#images=[]      
#res = ImageResult(k,size,champ)
#res.draw_shade([28,17])
#res.draw_shade([3,2])
#res.draw_shade([2,9])
#res.draw_shade([4,6])
#images.append(copy.copy(res.image))
#
#res.draw_case([2,9],False)
#res.draw_case([3,2],False)
#res.draw_case([4,6],False)
#res.draw_case([28,17],False)
#res.draw_case([23,16],True)
#images.append(copy.copy(res.image))
#
#res.draw_grid()
#
#images.append(res.image)    

def show_video(images, n) :
    title = "result" + str(n) +".gif"
    images[0].save(title,save_all=True, append_images=images[1:],duration=100, loop=False, optimize=True)