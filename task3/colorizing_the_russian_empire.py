import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

dir = './data/task3_colorizing'
save_dir = 'output/task3/data_output'
img_list = os.listdir(dir)


def process_image(img):
    h, w = img.shape
    # remove part of border
    img = img[int(h*0.01):int(h-h*0.02), int(w*0.02):int(w-w*0.02)]

    h, w = img.shape
    height = int(h/3)
    # split image into three channels
    blue = img[0:height, :]
    green = img[height:2*height, :]
    red = img[2*height:3*height, :]

    return blue, green, red , img

def edge_detection(channel):
    edges = cv2.Canny(channel, 100, 200)  # Canny 邊緣檢測的閾值可根據圖像調整
    return edges

def ncc(a, b):
    a=a-a.mean(axis=0)
    b=b-b.mean(axis=0)
    return np.sum(((a/np.linalg.norm(a)) * (b/np.linalg.norm(b))))

def allgn(a, b, t):
    max_ncc = -1
    max_ncc_shift = [0, 0]
    i_value = np.linspace(-t, t, 2*t, dtype=int)
    j_value = np.linspace(-t, t, 2*t, dtype=int)
    for i in i_value:
        for j in j_value:
            ncc_num = ncc(a, np.roll(b, [i, j], axis=(0,1)))
            if ncc_num > max_ncc:
                max_ncc = ncc_num
                max_ncc_shift = [i, j]
    return max_ncc_shift



if not os.path.exists(save_dir):
    os.makedirs(save_dir)
else:
    print(f"Directory {save_dir} already exists")

# data 包含'.jpg' 和 '.tif' 的檔案
for img in img_list:
    if img.endswith('.jpg'):
        print(img)
        name = img.split('.')[0]
        img = cv2.imread(os.path.join(dir, img), cv2.IMREAD_GRAYSCALE)

        blue,green,red,img = process_image(img)  


        blue_edges = edge_detection(blue)
        green_edges = edge_detection(green)
        red_edges = edge_detection(red)

        # 基於邊緣檢測結果進行對齊
        shift_red = allgn(blue_edges, red_edges, 20)
        shift_green = allgn(blue_edges, green_edges, 20)

        green = np.roll(green, shift_green, axis=(0, 1))
        red = np.roll(red, shift_red, axis=(0, 1))

        color_image = np.dstack([red, green, blue])
        color_image = color_image[int(img.shape[0]*0.02):int(img.shape[0]-img.shape[0]*0.05), int(img.shape[1]*0.05):int(img.shape[1]-img.shape[1]*0.05)]
        # plt.imshow(color_image)
        # plt.show()
        # 儲存之前將 RGB 轉換為 BGR (OpenCV)
        color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        # 保存彩色影像
        save_path = os.path.join(save_dir, name + '_color.jpg')
        result = cv2.imwrite(save_path, color_image_bgr)
        if not result:
            print(f"Failed to save image at {save_path}")
        else:
            print(f"Image saved at {save_path}")

    elif img.endswith('.tif'):
        print(img)
        name = img.split('.')[0]
        img = cv2.imread(os.path.join(dir, img), cv2.IMREAD_GRAYSCALE)

        subsample_rate = 10

        h, w = img.shape
        img = img[int(h * 0.01):int(h - h * 0.02), int(w * 0.02):int(w - w * 0.02)]  # the cutting ratio depends on image
        h, w = img.shape
        height = int(h/3)
        blue = img[0:height, :]
        green = img[height:2*height, :]
        red = img[2*height:3*height, :]

        # subsample image
        # Resize blue,green,red to smaller size before Align
        subsample_img= cv2.resize(img, (w//subsample_rate, h//subsample_rate), interpolation=cv2.INTER_CUBIC)  # better subsample result
        height = subsample_img.shape[0] // 3
        blue_ = subsample_img[0:height, :]
        green_ = subsample_img[height:2 * height, :]
        red_ = subsample_img[2 * height:3 * height, :]

        blue_edges = edge_detection(blue_)
        green_edges = edge_detection(green_)
        red_edges = edge_detection(red_)

        # get the shift value
        shift_green = allgn(blue_edges, green_edges, 15)
        shift_red = allgn(blue_edges, red_edges, 15)
    
        # shift green and red image
        green=np.roll(green,[i*subsample_rate for i in shift_green],axis=(0,1))
        red=np.roll(red,[i*subsample_rate for i in shift_red],axis=(0,1))

        color_image = np.dstack((red, green, blue))
        # cut the part of edge of the image
        color_image = color_image[int(h * 0.02):int(h - h * 0.05), int(w * 0.05):int(w - w * 0.05)]
        # plt.imshow(color_image)
        # plt.show()
        save_path = os.path.join(save_dir, name + '_color.jpg')
        # 儲存之前將 RGB 轉換為 BGR (OpenCV)
        color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        result = cv2.imwrite(save_path, color_image_bgr)
        if not result:
            print(f"Failed to save image at {save_path}")
        else:
            print(f"Image saved at {save_path}")
    else:
        print('Unknown file type:', img)
        continue