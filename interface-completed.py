#!/usr/bin/env python
# coding: utf-8

# In[51]:


from tkinter import *
from tkinter import filedialog
import os
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
from matplotlib import pyplot as plt
import math


# In[52]:


root = Tk()
root.geometry('500x500')
# low_pixel_intensity = None
# high_pixel_intensity = None 
# k_low_pixel_intensity = None
# k_high_pixel_intensity = None
# mu_b_low = None
# mu_b_high = None
# global low_pixel_intensity, high_pixel_intensity, k_low_pixel_intensity, k_high_pixel_intensity, mu_b_low,mu_b_high


# In[53]:


def bmd():
    K = math.log(k_low_pixel_intensity)/math.log(k_high_pixel_intensity)
    Numerator = (math.log(low_pixel_intensity)) - (K * math.log(high_pixel_intensity)) 
    Denominator = mu_b_low - (K * mu_b_high)
    M_b = Numerator/Denominator
    
    lbl = Label(root,text=str(M_b)).place(x = 150, y = 50)
    lbl.pack()


# In[54]:


def lpi(im):
    norm_img = np.zeros((800,800))
    final_image = cv2.normalize(im,  norm_img, 0, 255, cv2.NORM_MINMAX)
    low_image = Image.fromarray(final_image)
    low_cropped_img = low_image.crop((x_start,y_start,x_end,y_end))
    pixel_intensity = np.mean(low_cropped_img)
#     plt.imshow(low_cropped_img)
#     print(low_avg_pixel_intensity)
    return pixel_intensity
    


# In[55]:


def hpi(im):
    norm_img = np.zeros((800,800))
    final_image = cv2.normalize(im,  norm_img, 0, 255, cv2.NORM_MINMAX)
    high_image = Image.fromarray(final_image)
    high_cropped_img = high_image.crop((left,top,right,bottom))
    pixel_intensity = np.mean(high_cropped_img)
#     plt.imshow(low_cropped_img)
#     print(low_avg_pixel_intensity)
    return pixel_intensity
    


# In[56]:


def soft_lpi(im):
    norm_img = np.zeros((800,800))
    final_image = cv2.normalize(im,  norm_img, 0, 255, cv2.NORM_MINMAX)
    softl_image = Image.fromarray(final_image)
    softl_cropped_img = softl_image.crop((sleft,stop,sright,sbottom))
    pixel_intensity = np.mean(softl_cropped_img)
#     plt.imshow(low_cropped_img)
#     print(low_avg_pixel_intensity)
    return pixel_intensity
    


# In[57]:


def soft_hpi(im):
    norm_img = np.zeros((800,800))
    final_image = cv2.normalize(im,  norm_img, 0, 255, cv2.NORM_MINMAX)
    softh_image = Image.fromarray(final_image)
    softh_cropped_img = softh_image.crop((lefth,toph,righth,bottomh))
    pixel_intensity = np.mean(softh_cropped_img)
#     plt.imshow(low_cropped_img)
#     print(low_avg_pixel_intensity)
    return pixel_intensity
    


# In[58]:


# def high_pix(im):
#     norm_img = np.zeros((800,800))
#     final_image = cv2.normalize(im,  norm_img, 0, 255, cv2.NORM_MINMAX)
#     high_image = Image.fromarray(final_image)
#     high_cropped_img = high_image.crop((left,top,right,bottom))
#     high_avg_pixel_intensity = np.mean(high_cropped_img)
#     plt.imshow(high_cropped_img)
#     print(high_avg_pixel_intensity)
    


# In[59]:


# def soft_tissue_low(im):
#     norm_img = np.zeros((800,800))
#     final_image = cv2.normalize(im,  norm_img, 0, 255, cv2.NORM_MINMAX)
#     low_image = Image.fromarray(final_image)
#     low_cropped_img = low_image.crop((x_start,y_start,x_end,y_end))
#     low_avg_pixel_intensity = np.mean(low_cropped_img)
#     plt.imshow(low_cropped_img)
#     print(low_avg_pixel_intensity)
    


# In[60]:


# def soft_tissue_high(im):
#     norm_img = np.zeros((800,800))
#     final_image = cv2.normalize(im,  norm_img, 0, 255, cv2.NORM_MINMAX)
#     high_image = Image.fromarray(final_image)
#     high_cropped_img = low_image.crop((x_start,y_start,x_end,y_end))
#     low_avg_pixel_intensity = np.mean(low_cropped_img)
#     plt.imshow(low_cropped_img)
#     print(low_avg_pixel_intensity)
    


# In[61]:


def showimage():
    fln = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Image file", filetypes=[("ALL FILES", "*.*")])
    print(fln)
    if fln:
        image = cv2.imread(fln)
#         norm_img = np.zeros((800,800))
#         final_image = cv2.normalize(img,  norm_img, 0, 255, cv2.NORM_MINMAX)
#         image = Image.fromarray(final_image)
        
        im = image.copy()
        
#         cv2.imshow("slot", image)
#         cv2.waitKey(0)
#         cv2.destroyWindow("slot")
        cropping = False

        x_start, y_start, x_end, y_end = 0, 0, 0, 0
        global low_pixel_intensity, mu_b_low
        low_pixel_intensity = 0.0
        mu_b_low = 0.0

# image = cv2.imread('C:/Users/vbj/ffyp/Final-Year-Project/X-Rays/70_40.bmp')
        oriImage = image.copy()


        def mouse_crop(event, x, y, flags, param):
    # grab references to the global variables
            global x_start, y_start, x_end, y_end, cropping

            # if the left mouse button was DOWN, start RECORDING
            # (x, y) coordinates and indicate that cropping is being
            if event == cv2.EVENT_LBUTTONDOWN:
                x_start, y_start, x_end, y_end = x, y, x, y
                cropping = True

            # Mouse is Moving
            elif event == cv2.EVENT_MOUSEMOVE:
                if cropping == True:
                    x_end, y_end = x, y

            # if the left mouse button was released
            elif event == cv2.EVENT_LBUTTONUP:
                # record the ending (x, y) coordinates
                x_end, y_end = x, y
                cropping = False # cropping is finished

                refPoint = [(x_start, y_start), (x_end, y_end)]
                print(refPoint)
                if len(refPoint) == 2: #when two points were found
                    roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
                    cv2.imshow("Cropped", roi) 
                    cv2.imwrite("C:/Users/jashw/Final-Year-Project/X-Rays/r.png", roi)
                    img = Image.fromarray(roi)
                    img = PhotoImage(file="C:/Users/jashw/Final-Year-Project/X-Rays/r.png")
                    lbl = Label(root,image = img).place(x = 100, y = 50)
                    lbl.pack()

        cv2.namedWindow("image")
        cv2.setMouseCallback("image", mouse_crop)
        i = image.copy()

        if not cropping:
            cv2.imshow("image", image)

        elif cropping:
            cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
            cv2.imshow("image", i)
            
        low_pixel_intensity = lpi(im)
        print(low_pixel_intensity)
        if "70" in fln and "40" in fln:
            mu_b_low = 0.255
        elif "84" in fln and "60" in fln:
            mu_b_low = 0.214
            
        print(mu_b_low)
            
            
        
            
#         cv2.destroyAllWindows()

#             cv2.waitKey(1)

# close all open windows
#             cv2.destroyAllWindows()
#     img = Image.open(fln)
#     img = ImageTk.PhotoImage(roi)
#     lbl.configure(image=roi)
#     lbl.image = roi


# In[62]:


def highimage():
    fln = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Image file", filetypes=[("ALL FILES", "*.*")])
    print(fln)
    if fln:
        image = cv2.imread(fln)
#         norm_img = np.zeros((800,800))
#         final_image = cv2.normalize(img,  norm_img, 0, 255, cv2.NORM_MINMAX)
#         image = Image.fromarray(final_image)
        
        im = image.copy()
        
#         cv2.imshow("slot", image)
#         cv2.waitKey(0)
#         cv2.destroyWindow("slot")
        cropping_high = False
        global high_pixel_intensity, mu_b_high
        left, top, right, bottom = 0, 0, 0, 0
        high_pixel_intensity = 0.0
        mu_b_high = 0.0
# image = cv2.imread('C:/Users/vbj/ffyp/Final-Year-Project/X-Rays/70_40.bmp')
        oriImage = image.copy()


        def mouse_crop(event, x, y, flags, param):
    # grab references to the global variables
            global left, top, right, bottom, cropping_high

            # if the left mouse button was DOWN, start RECORDING
            # (x, y) coordinates and indicate that cropping is being
            if event == cv2.EVENT_LBUTTONDOWN:
                left, top, right, bottom = x, y, x, y
                cropping_high = True

            # Mouse is Moving
            elif event == cv2.EVENT_MOUSEMOVE:
                if cropping_high == True:
                    right, bottom = x, y

            # if the left mouse button was released
            elif event == cv2.EVENT_LBUTTONUP:
                # record the ending (x, y) coordinates
                right, bottom = x, y
                cropping_high = False # cropping is finished

                refPoint = [(left, top), (right, bottom)]
                print(refPoint)
                if len(refPoint) == 2: #when two points were found
                    roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
                    cv2.imshow("Cropped", roi) 
                    cv2.imwrite("C:/Users/jashw/Final-Year-Project/X-Rays/high.png", roi)
                    img = Image.fromarray(roi)
                    img = PhotoImage(file="C:/Users/jashw/Final-Year-Project/X-Rays/high.png")
                    lbl = Label(root,image = img).place(x = 150, y = 50)
                    lbl.pack()

        cv2.namedWindow("image")
        cv2.setMouseCallback("image", mouse_crop)
        i = image.copy()

        if not cropping_high:
            cv2.imshow("image", image)

        elif cropping_high:
            cv2.rectangle(i, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.imshow("image", i)
            
        high_pixel_intensity = hpi(im)
        print(high_pixel_intensity)
        if "111" in fln and "120" in fln:
            mu_b_high = 0.172
        elif "98" in fln and "100" in fln:
            mu_b_high = 0.189
        elif "92" in fln and "80" in fln:
            mu_b_high = 0.199
            
        print(mu_b_high)
            
            
        
            
#         cv2.destroyAllWindows()

#             cv2.waitKey(1)

# close all open windows
#             cv2.destroyAllWindows()
#     img = Image.open(fln)
#     img = ImageTk.PhotoImage(roi)
#     lbl.configure(image=roi)
#     lbl.image = roi


# In[63]:


def lowsoft():
    fln = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Image file", filetypes=[("ALL FILES", "*.*")])
    print(fln)
    if fln:
        image = cv2.imread(fln)
#         norm_img = np.zeros((800,800))
#         final_image = cv2.normalize(img,  norm_img, 0, 255, cv2.NORM_MINMAX)
#         image = Image.fromarray(final_image)
        
        im = image.copy()
        
#         cv2.imshow("slot", image)
#         cv2.waitKey(0)
#         cv2.destroyWindow("slot")
        scropping = False
        global k_low_pixel_intensity
        sleft, stop, sright, sbottom = 0, 0, 0, 0
        k_low_pixel_intensity = 0.0 

# image = cv2.imread('C:/Users/vbj/ffyp/Final-Year-Project/X-Rays/70_40.bmp')
        oriImage = image.copy()


        def mouse_crop(event, x, y, flags, param):
    # grab references to the global variables
            global sleft, stop, sright, sbottom, scropping
    
            # if the left mouse button was DOWN, start RECORDING
            # (x, y) coordinates and indicate that cropping is being
            if event == cv2.EVENT_LBUTTONDOWN:
                sleft, stop, sright, sbottom = x, y, x, y
                scropping = True

            # Mouse is Moving
            elif event == cv2.EVENT_MOUSEMOVE:
                if scropping == True:
                    sright, sbottom = x, y

            # if the left mouse button was released
            elif event == cv2.EVENT_LBUTTONUP:
                # record the ending (x, y) coordinates
                sright, sbottom = x, y
                scropping = False # cropping is finished

                refPoint = [(sleft, stop), (sright, sbottom)]
                print(refPoint)
                if len(refPoint) == 2: #when two points were found
                    roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
                    cv2.imshow("Cropped", roi) 
                    cv2.imwrite("C:/Users/jashw/Final-Year-Project/X-Rays/slow.png", roi)
                    img = Image.fromarray(roi)
                    img = PhotoImage(file="C:/Users/jashw/Final-Year-Project/X-Rays/slow.png")
                    lbl = Label(root,image = img).place(x = 150, y = 50)
                    lbl.pack()

        cv2.namedWindow("image")
        cv2.setMouseCallback("image", mouse_crop)
        i = image.copy()

        if not cropping_high:
            cv2.imshow("image", image)

        elif cropping_high:
            cv2.rectangle(i, (sleft, stop), (sright, sbottom), (255, 0, 0), 2)
            cv2.imshow("image", i)
        
        k_low_pixel_intensity = soft_lpi(im)
        print(k_low_pixel_intensity)
#         if 70 in fln and 40 in fln:
#             mu_b_low = 0.192
#         elif 84 in fln and 60 in fln:
#             mu_b_low = 0.1795
#         elif 92 in fln and 80 in fln:
#             mu_b_low = 
#         high_pix(im)
            
        
            
#         cv2.destroyAllWindows()

#             cv2.waitKey(1)

# close all open windows
#             cv2.destroyAllWindows()
#     img = Image.open(fln)
#     img = ImageTk.PhotoImage(roi)
#     lbl.configure(image=roi)
#     lbl.image = roi


# In[64]:


def highsoft():
    fln = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Image file", filetypes=[("ALL FILES", "*.*")])
    print(fln)
    if fln:
        image = cv2.imread(fln)
#         norm_img = np.zeros((800,800))
#         final_image = cv2.normalize(img,  norm_img, 0, 255, cv2.NORM_MINMAX)
#         image = Image.fromarray(final_image)
        
        im = image.copy()
        
#         cv2.imshow("slot", image)
#         cv2.waitKey(0)
#         cv2.destroyWindow("slot")
        scropping_high = False
        global k_high_pixel_intensity
        lefth, toph, righth, bottomh = 0, 0, 0, 0
        k_high_pixel_intensity = 0.0
# image = cv2.imread('C:/Users/vbj/ffyp/Final-Year-Project/X-Rays/70_40.bmp')
        oriImage = image.copy()


        def mouse_crop(event, x, y, flags, param):
    # grab references to the global variables
            global lefth, toph, righth, bottomh, scropping_high

            # if the left mouse button was DOWN, start RECORDING
            # (x, y) coordinates and indicate that cropping is being
            if event == cv2.EVENT_LBUTTONDOWN:
                lefth, toph, righth, bottomh = x, y, x, y
                scropping_high = True

            # Mouse is Moving
            elif event == cv2.EVENT_MOUSEMOVE:
                if scropping_high == True:
                    righth, bottomh = x, y

            # if the left mouse button was released
            elif event == cv2.EVENT_LBUTTONUP:
                # record the ending (x, y) coordinates
                righth, bottomh = x, y
                scropping_high = False # cropping is finished

                refPoint = [(lefth, toph), (righth, bottomh)]
                print(refPoint)
                if len(refPoint) == 2: #when two points were found
                    roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
                    cv2.imshow("Cropped", roi) 
                    cv2.imwrite("C:/Users/jashw/Final-Year-Project/X-Rays/shigh.png", roi)
                    img = Image.fromarray(roi)
                    img = PhotoImage(file="C:/Users/jashw/Final-Year-Project/X-Rays/shigh.png")
                    lbl = Label(root,image = img).place(x = 150, y = 50)
                    lbl.pack()

        cv2.namedWindow("image")
        cv2.setMouseCallback("image", mouse_crop)
        i = image.copy()

        if not scropping_high:
            cv2.imshow("image", image)

        elif scropping_high:
            cv2.rectangle(i, (lefth, toph), (righth, bottomh), (255, 0, 0), 2)
            cv2.imshow("image", i)
        
        
        k_high_pixel_intensity = soft_hpi(im)
        print(k_high_pixel_intensity)
#         if 111 in fln and 120 in fln:
#             mu_b_low = 0.163 
#         elif 98 in fln and 100 in fln:
#             mu_b_low = 
#         high_pix(im)
            
        
            
#         cv2.destroyAllWindows()

#             cv2.waitKey(1)

# close all open windows
#             cv2.destroyAllWindows()
#     img = Image.open(fln)
#     img = ImageTk.PhotoImage(roi)
#     lbl.configure(image=roi)
#     lbl.image = roi


# In[65]:



# import cv2
# import numpy as np


# cropping = False

# x_start, y_start, x_end, y_end = 0, 0, 0, 0

# image = cv2.imread('C:/Users/vbj/ffyp/Final-Year-Project/X-Rays/70_40.bmp')
# oriImage = image.copy()


# def mouse_crop(event, x, y, flags, param):
#     # grab references to the global variables
#     global x_start, y_start, x_end, y_end, cropping

#     # if the left mouse button was DOWN, start RECORDING
#     # (x, y) coordinates and indicate that cropping is being
#     if event == cv2.EVENT_LBUTTONDOWN:
#         x_start, y_start, x_end, y_end = x, y, x, y
#         cropping = True

#     # Mouse is Moving
#     elif event == cv2.EVENT_MOUSEMOVE:
#         if cropping == True:
#             x_end, y_end = x, y

#     # if the left mouse button was released
#     elif event == cv2.EVENT_LBUTTONUP:
#         # record the ending (x, y) coordinates
#         x_end, y_end = x, y
#         cropping = False # cropping is finished

#         refPoint = [(x_start, y_start), (x_end, y_end)]
#         print(refPoint)
#         if len(refPoint) == 2: #when two points were found
#             roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
#             cv2.imshow("Cropped", roi)

# cv2.namedWindow("image")
# cv2.setMouseCallback("image", mouse_crop)

# while True:

#     i = image.copy()

#     if not cropping:
#         cv2.imshow("image", image)

#     elif cropping:
#         cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
#         cv2.imshow("image", i)

#     cv2.waitKey(1)

# # close all open windows
# cv2.destroyAllWindows()


# In[ ]:


frame = Frame(root)
frame.pack(side=BOTTOM, padx=15, pady = 15)

lbl = Label(root)
lbl.pack()

btn = Button(frame,text="Low-Image", command=showimage)
btn.pack(side=tk.LEFT)

btn2 = Button(frame,text="High-Image", command=highimage)
btn2.pack(side=tk.LEFT,padx = 10)

btn3 = Button(frame,text="Low-Soft-tissue", command=lowsoft)
btn3.pack(side=tk.LEFT)

btn4 = Button(frame,text="High-Soft-tissue", command=highsoft)
btn4.pack(side=tk.LEFT,padx = 10)


btn5 = Button(frame,text="BMD", command=bmd)
btn5.pack(side=tk.LEFT,padx = 10)

# btn2 = Button(frame,text = "Exit", command=lambda: exit())
# btn2.pack(side=tk.LEFT,padx = 10)

# btn3 = Button(frame,text = "Save")
# btn3.pack(side=tk.LEFT,padx = 10)

root.title("BMD")
root.geometry("300x350")
root.mainloop()


# In[ ]:


# print(x_start,y_start,x_end,y_end)
# print(left,right,top,bottom)
print(sleft,sright,stop,sbottom)
# print(lefth,righth,toph,bottomh)


# In[ ]:


im


# In[ ]:





# In[ ]:


print(low_pixel_intensity)
# high_pixel_intensity  
# k_low_pixel_intensity 
# k_high_pixel_intensity
# mu_b_low
# mu_b_high
print(high_pixel_intensity)
print(k_low_pixel_intensity)
print(k_high_pixel_intensity)
print(mu_b_low)
print(mu_b_high)


# In[ ]:


print(low_pixel_intensity)


# In[ ]:


# def bmd():
#     K = math.log(k_low_pixel_intensity)/math.log(k_high_pixel_intensity)
#     Numerator = (math.log(low_pixel_intensity)) - (K * math.log(high_pixel_intensity)) 
#     Denominator = mu_b_low - (K * mu_b_high)
#     M_b = Numerator/Denominator
    
#     lbl = Label(root,text=str(M_b)).place(x = 150, y = 50)
#     lbl.pack()


# In[ ]:


bmd()


# In[ ]:




