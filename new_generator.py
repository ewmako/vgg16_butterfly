import time
import tensorflow as tf
import numpy as np
import cv2
import os
from scipy.misc import imread, imresize

path_imgs = 'leedsbutterfly/train_data'
train_final = 'leedsbutterfly/train_final'
path_masks = 'leedsbutterfly/segmentations'
log_base = 'logs'

if __name__ == '__main__':
    
    filenames = os.listdir(path_imgs)
    np.random.shuffle(filenames)
    
    for filename in filenames:
        if not filename.endswith("png"):
            continue
        for mask_name in os.listdir(path_masks):
            if mask_name == filename[:-4] + "_seg0.png":
                mask = cv2.imread(os.path.join(path_masks, mask_name), 0)
        img = cv2.imread(os.path.join(path_imgs, filename), 1)
        h, w = img.shape[:2]
        max_size = max(h, w)
           
        # Insert img into new squared
        square = np.zeros((3 * max_size, 3 * max_size, 3), np.uint8)
        square[(3 * max_size - h) / 2:(3 * max_size - h) / 2 + h,
               (3 * max_size - w) / 2:(3 * max_size - w) / 2 + w] = img
        
        butterfly = np.argwhere(mask == 255)
        
        y_max = butterfly[np.argmax(butterfly[:, 0]), 0]
        y_min = butterfly[np.argmin(butterfly[:, 0]), 0]
        x_max = butterfly[np.argmax(butterfly[:, 1]), 1]
        x_min = butterfly[np.argmin(butterfly[:, 1]), 1]
        
        y_c = (y_max - y_min) / 2 + (3 * max_size - h) / 2 + y_min
        x_c = (x_max - x_min) / 2 + (3 * max_size - w) / 2 + x_min

        max_size_butterfly = max(y_max - y_min, x_max - x_min)
        
        #print 'Y_c: %05d X_c: %05d Max_size_butterfly: %05d' % (y_c, x_c, max_size_butterfly)
        
        y1 = y_c - max_size_butterfly / 2
        y2 = y_c + max_size_butterfly / 2
        x1 = x_c - max_size_butterfly / 2
        x2 = x_c + max_size_butterfly / 2
        
        #print 'Y1: %05d Y2: %05d X1: %05d X2: %05d' % (y1, y2, x1, x2)
        
        crop_img = square[y1:y2, x1:x2]
        
        resized_img = imresize(crop_img, (224, 224))
        
        # write resized original image
        cv2.imwrite(os.path.join(train_final , filename), resized_img)
        
        # rotate
        M = cv2.getRotationMatrix2D((112, 112), 15, 1)
        rot_img15 = cv2.warpAffine(resized_img.copy(), M, (224, 224))
        M = cv2.getRotationMatrix2D((112, 112), 345, 1)
        rot_img345 = cv2.warpAffine(resized_img.copy(), M, (224, 224))
        M = cv2.getRotationMatrix2D((112, 112), 30, 1)
        rot_img30 = cv2.warpAffine(resized_img.copy(), M, (224, 224))
        M = cv2.getRotationMatrix2D((112, 112), 330, 1)
        rot_img330 = cv2.warpAffine(resized_img.copy(), M, (224, 224))
        
        # write rotate
        cv2.imwrite(os.path.join(train_final , filename[:-4] + "f.png"), rot_img15)
        cv2.imwrite(os.path.join(train_final , filename[:-4] + "g.png"), rot_img345)
        cv2.imwrite(os.path.join(train_final , filename[:-4] + "h.png"), rot_img30)
        cv2.imwrite(os.path.join(train_final , filename[:-4] + "i.png"), rot_img330)
        
        # mirror image
        mirror_img = cv2.flip(resized_img.copy(), 1)
        
        # write mirror image
        cv2.imwrite(os.path.join(train_final , filename[:-4] + "a.png"), mirror_img)
