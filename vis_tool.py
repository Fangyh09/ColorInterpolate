import cv2
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
# import boxx
#FFFFFF
# x: #068683, -x: #F97F83
# y: #840391, -y: #82F576
# z: #7A790A, -z: #7C78F7
X_COLOR = [[[[6,134,131]]], [[[249,127,131]]]]
Y_COLOR = [[[[132,3,145]]], [[[130,245,118]]]]
Z_COLOR = [[[[122,121,10]]], [[[124,120,247]]]]

# X_COLOR = [[[[0,0,0]]], [[[249,0,0]]]]
# Y_COLOR = [[[[0,0,0]]], [[[0,245,0]]]]
# Z_COLOR = [[[[0,0,0]]], [[[0,0,247]]]]
# 125, 120, 135`
# X_COLOR = [[[[-120,0,0]]], [[[120,0,0]]]]
# Y_COLOR = [[[[0,-120,0]]], [[[0,120,0]]]]
# Z_COLOR = [[[[0,0,-120]]], [[[0,0,120]]]]


X_COLOR = np.array(X_COLOR).astype(np.uint8)
Y_COLOR = np.array(Y_COLOR).astype(np.uint8)
Z_COLOR = np.array(Z_COLOR).astype(np.uint8)
# X_COLOR = np.array(X_COLOR).astype`(np.float32)
# Y_COLOR = np.array(Y_COLOR).astype(np.float32)
# Z_COLOR = np.array(Z_COLOR).astype(np.float32)

def bgr2hsv(bgr_img):
    # import pdb
    # pdb.set_trace()
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    return hsv_img
    
def hsv2bgr(hsv_img):
    # import pdb
    # pdb.set_trace()
    bgr_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return bgr_img


def rgb2hsv(rgb_img):
    # import pdb
    # pdb.set_trace()
    bgr_img = rgb_img[:,:,::-1]
    hsv_img = bgr2hsv(bgr_img)
    # rev_hsv_img = hsv_img.transpose(hsv_img, (2,1,0))
    return hsv_img

def hsv2rgb(hsv_img):
    bgr_img = hsv2bgr(hsv_img)
    # rgb_img = np.transpose(bgr_img, (2,1,0))
    rgb_img = bgr_img[:,:,::-1]
    return rgb_img

def interplot_color(c1, c2, c2_weight):
    c2_weight = np.clip(c2_weight, a_min=0, a_max=1)
    assert c2_weight <= 1.0
    h1 = rgb2hsv(c1)
    h2 = rgb2hsv(c2)
    h = (h2 - h1) * c2_weight + h1
    h[0] = int(h[0])
    h[1] = int(h[1])
    h[2] = int(h[2])
    h = np.clip(h, a_min=0, a_max=255)
    h = h.astype(np.uint8)
    return h

def clip0_255(img):
    # logger.debug(img)
    clip_img = np.clip(img, a_min=0, a_max=255).astype(np.uint8)
    # logger.debug(clip_img)
    return clip_img
  

def norm_color2(w1, w2, w3):            
    H, W = w1.shape
    x_neg = rgb2hsv(X_COLOR[1]); x_pos = rgb2hsv(X_COLOR[0])
    y_neg = rgb2hsv(Y_COLOR[1]); y_pos = rgb2hsv(Y_COLOR[0])
    z_neg = rgb2hsv(Z_COLOR[1]); z_pos = rgb2hsv(Z_COLOR[0])
    x_neg = x_neg.astype(np.float32)
    x_pos = x_pos.astype(np.float32)
    y_neg = y_neg.astype(np.float32)
    y_pos = y_pos.astype(np.float32)
    z_neg = z_neg.astype(np.float32)
    z_pos = z_pos.astype(np.float32)
    
    # w1 = (w1 + 1) / 2.0; w2 = (w2 + 1) / 2.0; w3 = (w3 + 1) / 2.0; 
    w1 = np.reshape(w1, (H, W, 1))
    w2 = np.reshape(w2, (H, W, 1))
    w3 = np.reshape(w3, (H, W, 1))
    
    hsv = x_neg * w1 + x_neg + \
            y_neg * w1 + y_neg  + \
            z_neg * w3 + z_neg 

    # hsv = np.clip() 
    hsv = clip0_255(hsv)
    rgb_img = hsv2rgb(hsv)
    rgb_img = clip0_255(rgb_img)       
    return rgb_img

    
                         
         
def norm_color(w1, w2, w3):
    H, W = w1.shape
    x_neg = rgb2hsv(X_COLOR[1]); x_pos = rgb2hsv(X_COLOR[0])
    y_neg = rgb2hsv(Y_COLOR[1]); y_pos = rgb2hsv(Y_COLOR[0])
    z_neg = rgb2hsv(Z_COLOR[1]); z_pos = rgb2hsv(Z_COLOR[0])
    x_neg = x_neg.astype(np.float32)
    x_pos = x_pos.astype(np.float32)
    y_neg = y_neg.astype(np.float32)
    y_pos = y_pos.astype(np.float32)
    z_neg = z_neg.astype(np.float32)
    z_pos = z_pos.astype(np.float32)
    # logger.info(np.max(w1), np.min)
    
    # assert (w1 >= -1).all() and (w1 <= 1).all()
    w1 = (w1 + 1) / 2.0; w2 = (w2 + 1) / 2.0; w3 = (w3 + 1) / 2.0; 

    w1 = np.reshape(w1, (H, W, 1))
    w2 = np.reshape(w2, (H, W, 1))
    w3 = np.reshape(w3, (H, W, 1))
    
    tot = np.abs(w1 * 2 - 1) + np.abs(w2 * 2 - 1) + np.abs(w3 * 2 - 1)
    # logger.debug(tot)
    tot = np.clip(tot, a_min=1e-6, a_max=3.0)
    
    norm_w1 = np.abs(w1 * 2 - 1) / tot
    norm_w2 = np.abs(w2 * 2 - 1) / tot
    norm_w3 = np.abs(w3 * 2 - 1) / tot
    
    
    # import pdb
    # pdb.set_trace()
    hsv = ((x_pos - x_neg) * w1 + x_neg) * norm_w1 + \
            ((y_pos - y_neg) * w2 + y_neg) * norm_w2 + \
            ((z_pos - z_neg) * w3 + z_neg) * norm_w3
    # hsv = np.clip()
    # logger.debug(hsv)
    hsv = clip0_255(hsv)
    rgb_img = hsv2rgb(hsv)
    rgb_img = clip0_255(rgb_img)
    # rgb_img = np.clip(rgb_img, a_min=0, a_max=255)
    return rgb_img
        

if __name__ == "__main__":  
    # color = np.random.randn(100, 100, 3).astype(np.uint8)
    # hsv = rgb2hsv(color)
    # import pdb
    # pdb.set_trace()
    w1 = np.random.rand(100, 100)
    w2 = np.random.rand(100, 100)
    w3 = np.random.rand(100, 100)
    # w1 = w2 = w3 = np.array([[1]]).astype(np.float32)
    m1 = norm_color(w1, w2, w3)
    # logger.debug(w1)
    # logger.debug(w2)
    # logger.debug(w3)
    # logger.debug(m1)
    # logger.debug(m1.shape)
    plt.imsave("m1.png", m1)
    pass
    # blues = [[77, 77, 255], [102, 102, 255], [128, 128, 255]]
    # img0 = np.ones((2,2),dtype=np.uint8)
    # bgr_img = cv2.cvtColor(img0,cv2.COLOR_GRAY2BGR)
    # blues_hsv = []
    # for i in blues:
    #     r,g,b = (i[0],i[1],i[2])
    #     bgr_img[:,:,:] = (b,g,r) 
    #     HSV = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    #     blues_hsv.append(HSV[1,1,:].tolist())
    #     print(blues_hsv)
