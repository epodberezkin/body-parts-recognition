import math
import random

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from sklearn.ensemble import RandomForestRegressor
from scipy import ndimage
from sklearn.model_selection import train_test_split
import sys
import time
import makeFeatures
sys.path.append("data/final_task")
from baseml.data import StandardDataset

class RFRModel:
    
    SAVE = True
    FEATURES = 200
    TRAIN_PIX = 2000
    TRAIN_PIX_PART = 333
    fx = 554.25
    fy = 520.6
    cx = 320
    cy = 240
    SIZE_X = 640
    SIZE_Y = 480
    SIZE = min(SIZE_X, SIZE_Y)
    JOINTS = ["head", "left_wrist", "right_wrist"]
    score_limit = 0.5
    waterline = 360
    bg_depth = 8160
    mean_shift_window_size = 10

    
    def __init__(self, train=False):
        self.clf, self.features, self.data_test = self.getModel(train)
        
    def getModel(self, train=False):
        if train:
            return self._trainModel(RFRModel.FEATURES, RFRModel.SAVE)
        else:
            return joblib.load("model.pkl")

    def _trainModel(self, featureCount, save):
        self.clf = RandomForestRegressor(n_estimators=20, n_jobs=-1, max_depth=15, verbose=0)
        ds = StandardDataset("data/final_task/public_dataset")
        features = RFRModel._generateFeatures(featureCount)
        data_ds = []
        for group in ds.groups:
            data_ds.extend(ds >> group)

        data_train, data_test = train_test_split(data_ds, test_size=0.1, random_state=42)

        n_trains = len(data_train)
        n_points = len(RFRModel.JOINTS)
        X = np.zeros((n_trains * RFRModel.TRAIN_PIX, RFRModel.FEATURES), dtype=np.int32)
        Y = np.zeros((n_trains * RFRModel.TRAIN_PIX, n_points), dtype=np.float32)
        #variable for skipping bad frames
        delta = 0
        for n, frame in enumerate(data_train):
            res_train = RFRModel._get_train(features, frame)
            if res_train is None:
                delta += 1
                continue
            img_features, img_targets = res_train
            X[(n - delta) * RFRModel.TRAIN_PIX:(n - delta + 1) * RFRModel.TRAIN_PIX, :] = img_features
            Y[(n - delta) * RFRModel.TRAIN_PIX:(n - delta + 1) * RFRModel.TRAIN_PIX, :] = img_targets
        # Fit model
        self.clf.fit(X[:(n_trains - delta) * RFRModel.TRAIN_PIX], Y[:(n_trains - delta) * RFRModel.TRAIN_PIX])
        model = (self.clf, features, data_test)

        if save:
            joblib.dump(model, "model.pkl")
        return model

    @staticmethod
    def _get_train(features, frame):
        
        # get inversed mask
        try:
            inv_mask = cv2.bitwise_not(frame['segment_id'])
            # substruct background
            depth_without_bg = frame['segment_id'] / 255 * frame['depth']
            # set background to constant far value
            depth = depth_without_bg + inv_mask.astype(np.uint16) * 32
            # get targets
            head = frame['objects'][0]['skeleton']['head']['real']
            wrist_l = frame['objects'][0]['skeleton']['left_wrist']['real']
            wrist_r = frame['objects'][0]['skeleton']['right_wrist']['real']
        except KeyError:
            return None

        #convert to 0:255 range
        depth_sp = np.uint8(depth_without_bg / np.max(depth_without_bg) * 255)
        #calculate SuperPixel labels
        sp_slic = cv2.ximgproc.createSuperpixelSLIC(depth_sp, cv2.ximgproc.SLICO, 32, 10.0)
        sp_slic.iterate()
        lb = sp_slic.getLabels()

        # get target`s u,v,z
        head_proj = RFRModel._xproj(*head)
        wrist_l_proj = RFRModel._xproj(*wrist_l)
        wrist_r_proj = RFRModel._xproj(*wrist_r)

        # get target`s superpixel labels
        head_label = lb[np.uint16(head_proj[1]), np.uint16(head_proj[0])]
        wrist_l_label = lb[np.uint16(wrist_l_proj[1]), np.uint16(wrist_l_proj[0])]
        wrist_r_label = lb[np.uint16(wrist_r_proj[1]), np.uint16(wrist_r_proj[0])]

        # get target`s mask and mask`s indexes
        mask_h = np.zeros(shape=lb.shape)
        mask_wl = np.zeros(shape=lb.shape)
        mask_wr = np.zeros(shape=lb.shape)

        head_indx = (lb == head_label)
        mask_h[head_indx] = 1
        mask_h_indx = np.transpose(np.where(head_indx))

        wrist_l_indx = (lb == wrist_l_label)
        mask_wl[wrist_l_indx] = 1
        mask_wl_indx = np.transpose(np.where(wrist_l_indx))

        wrist_r_indx = (lb == wrist_r_label)
        mask_wr[wrist_r_indx] = 1
        mask_wr_indx = np.transpose(np.where(wrist_r_indx))

        # sampling targets? calculate positive and negative
        size_s = min(mask_h_indx.shape[0], mask_wl_indx.shape[0], mask_wr_indx.shape[0])
        if size_s > RFRModel.TRAIN_PIX_PART:
            size_s = RFRModel.TRAIN_PIX_PART
        size_neg = RFRModel.TRAIN_PIX - size_s * 3
        
        #get size_s random coordinates of each body part
        mh_r = mask_h_indx[np.random.randint(mask_h_indx.shape[0], size=size_s), :]
        ml_r = mask_wl_indx[np.random.randint(mask_wl_indx.shape[0], size=size_s), :]
        mr_r = mask_wr_indx[np.random.randint(mask_wr_indx.shape[0], size=size_s), :]
        #add size_neg random coordinates from the whole depth image
        size_neg_y = np.random.randint(RFRModel.SIZE_Y - 1, size=(size_neg,))
        size_neg_x = np.random.randint(RFRModel.SIZE_X - 1, size=(size_neg,))
        neg_sample = np.stack((size_neg_y, size_neg_x)).transpose()
        
        #prepare TRAIN_PIX shuffle coordinates
        sample_indx = np.concatenate((mh_r, ml_r, mr_r, neg_sample))
        np.random.shuffle(sample_indx)
        sample_indx = sample_indx.transpose()
        
        #prepare Y_train - one-hot like encoding, 000 means negative sample, 100-head, 010-wl, 001-rl
        mask_h_sampled = mask_h[sample_indx[0], sample_indx[1]]
        mask_wl_sampled = mask_wl[sample_indx[0], sample_indx[1]]
        mask_wr_sampled = mask_wr[sample_indx[0], sample_indx[1]]
        
        Y_train = np.zeros((RFRModel.TRAIN_PIX, len(RFRModel.JOINTS)), dtype=np.float32)
        
        Y_train[:, 0] = mask_h_sampled
        Y_train[:, 1] = mask_wl_sampled
        Y_train[:, 2] = mask_wr_sampled

        X_train = np.zeros((RFRModel.TRAIN_PIX, RFRModel.FEATURES), dtype=np.int32)
        X_image = np.asarray(makeFeatures.make_features(depth.astype(np.int32),
                                                        np.asarray(features,dtype=np.int32),
                                                        0,0,RFRModel.SIZE_Y,RFRModel.SIZE_X)) 
        X_train = X_image[sample_indx[0] * RFRModel.SIZE_X + sample_indx[1]]
        return X_train, Y_train

    @staticmethod
    def _generateFeatures(count):
        return list(map(RFRModel._get_randvec, [None] * count))

    @staticmethod
    def _get_randvec(arg):
        if random.random() > 0.5:
            return RFRModel._get_randoffset(RFRModel.SIZE / 2), RFRModel._get_randoffset(RFRModel.SIZE / 2)
        else:
            return RFRModel._get_randoffset(RFRModel.SIZE / 2), np.array([0, 0])

    @staticmethod
    def _get_randoffset(sd):
        return np.array([random.uniform(-sd, sd), random.uniform(-sd, sd)]).astype(np.int32)

    @staticmethod
    def _select(height, width, depth, offset, C):
        if np.array_equal(offset, [0, 0]):
            return depth
        out = np.full((height, width), C, dtype=np.int32)
        target_y_from = 0 if offset[0] > 0 else -offset[0]
        target_y_to = height if offset[0] < 0 else height - offset[0]
        target_x_from = 0 if offset[1] > 0 else -offset[1]
        target_x_to = width if offset[1] < 0 else width - offset[1]
        source_y_from = offset[0] if offset[0] > 0 else 0
        source_y_to = height if offset[0] > 0 else height + offset[0]
        source_x_from = offset[1] if offset[1] > 0 else 0
        source_x_to = width if offset[1] > 0 else width + offset[1]
        out[target_y_from:target_y_to, target_x_from:target_x_to] = \
            depth[source_y_from:source_y_to, source_x_from:source_x_to]
        return out

    @staticmethod
    def _xreal(u, v, z):
        x = (u - RFRModel.cx) / RFRModel.fx * z
        y = (RFRModel.cy - v) / RFRModel.fy * z
        return np.array([x, y, z])

    @staticmethod
    def _xproj(x, y, z):
        u = x * RFRModel.fx / z + RFRModel.cx
        v = -y * RFRModel.fy / z + RFRModel.cy
        return np.array([u, v, z])
    
    def process(self, data):
        # unpack current frame data
        depth, mask, rgb = data
        inv_mask = cv2.bitwise_not(mask)
        # substruct background
        depth_without_bg = mask / 255 * depth
        # set background to constant far value
        depth = depth_without_bg + inv_mask.astype(np.uint16) * 32
        ymin,xmin = np.min(np.nonzero(mask),axis=1)
        ymax,xmax = np.max(np.nonzero(mask),axis=1)
        X_image = np.asarray(makeFeatures.make_features(depth.astype(np.int32),
                                                        np.asarray(self.features,dtype=np.int32),
                                                        ymin,xmin,RFRModel.waterline,xmax+1))
        indx = np.nonzero(mask[:RFRModel.waterline,:])
        #get test values for all non-zero mask pixels 
        X_test = X_image[indx[0] * RFRModel.SIZE_X + indx[1]]
        # points prediction
        Y_test = self.clf.predict(X_test)
        # treshold
        Y_test = np.where(Y_test > RFRModel.score_limit, Y_test, 0.0)
        # background predictions is 0
        Y_image = np.zeros((RFRModel.SIZE_Y, RFRModel.SIZE_X, len(RFRModel.JOINTS)))
        Y_image[indx] = Y_test
        # Y_image = Y_image.reshape((RFRModel.SIZE_Y, RFRModel.SIZE_X, len(RFRModel.JOINTS)))
        #testing plots
        #fig, axes = plt.subplots(1,4,figsize=(15,15))
        #axes[0].imshow(depth,cmap='gray')
        predicted_coord = np.zeros((3, 3), dtype=np.float32)
        for i in range(len(RFRModel.JOINTS)):
            image = Y_image[:, :, i]
            #coord = RFRModel._get_coordinates(image)
            erosed_image = ndimage.grey_erosion(image, size=(7,7))
            if np.max(erosed_image) == 0:
                coord = (0, 0)
            else:
                labeled_image, number_of_objects = ndimage.label(erosed_image)
                centroids = ndimage.center_of_mass(erosed_image, labeled_image, range(1, number_of_objects+1))
                sums = RFRModel._weighted_sum(erosed_image, RFRModel.mean_shift_window_size, centroids)
                ms = np.argmax(sums)
                coord = RFRModel._mean_shift(erosed_image, RFRModel.mean_shift_window_size, centroids[ms])
            z_value = RFRModel._mean_z_coord(depth, image, coord)
            z_shift = 130 if (i == 0) else 20
            xreal_coord = RFRModel._xreal(coord[1], coord[0], z_value + z_shift)
            predicted_coord[i, :] = xreal_coord
            #image[coord[0]-5:coord[0]+5,coord[1]-5:coord[1]+5] = 0.5
            #axes[i+1].imshow(image,cmap='gray')
        #plt.show()
        return predicted_coord
    
    @staticmethod
    def _weighted_sum(img, ms_ws, source_points):
        gk = RFRModel._gkern(ms_ws * 2 + 1, 3)
        padded_img = np.pad(img, ms_ws, mode='edge')
        sum = []
        for new_point in source_points:
            new_point = [int(math.floor(point)) + ms_ws for point in new_point]
            M_density = np.multiply(gk, padded_img[new_point[0] - ms_ws:new_point[0] + ms_ws + 1,
                                        new_point[1] - ms_ws:new_point[1] + ms_ws + 1])
            sum.append(np.sum(M_density))
        return sum

    @staticmethod
    def _gkern(kernel_len=21, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        interval = (2 * nsig + 1.) / kernel_len
        x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernel_len + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    @staticmethod
    def _get_coordinates(heat_map):
        if np.max(heat_map[:RFRModel.waterline, :]) == 0:
            ms = (0, 0)
        else:
            start = ndimage.measurements.center_of_mass(heat_map[:RFRModel.waterline, :])
            ms = RFRModel._mean_shift(heat_map, 10, start)
        return ms

    @staticmethod
    def _mean_shift(img, ms_ws, source_point):
        new_point = np.float64(source_point) + ms_ws
        Mx = [10, 10]
        padded_img = np.pad(img, ms_ws, mode='edge')
        gk = RFRModel._gkern(ms_ws * 2 + 1)
        while np.linalg.norm(Mx) > 1:
            shift_window = padded_img[int(math.floor(new_point[0])) - ms_ws:int(math.floor(new_point[0])) + ms_ws + 1,
                                    int(math.floor(new_point[1])) - ms_ws:int(math.floor(new_point[1])) + ms_ws + 1]
            if np.max(shift_window) == 0:
                Mx = (0, 0)
            else:
                M_density = np.multiply(gk, shift_window)
                Mx = np.subtract(ndimage.measurements.center_of_mass(M_density), (ms_ws + 0.5, ms_ws + 0.5))
            new_point += Mx
        return new_point.astype(np.uint16) - ms_ws


    @staticmethod
    def _mean_z_coord(depth, img, coord):
        shift = 4
        # indexes of non-background pixels in 9x9 area
        x0 = 0 if coord[0] - shift < 0 else coord[0] - shift
        y0 = 0 if coord[1] - shift < 0 else coord[1] - shift
        x1 = RFRModel.SIZE_X - 1 if coord[0] + shift + 1 > RFRModel.SIZE_X - 1 else coord[0] + shift + 1
        y1 = RFRModel.SIZE_Y - 1 if coord[1] + shift + 1 > RFRModel.SIZE_Y - 1 else coord[1] + shift + 1
        non_bg_indx = np.where(img[x0:x1, y0:y1] != 0)
        depth_slice = depth[x0:x1, y0:y1]
        z_values = depth_slice[non_bg_indx]
        z_value = np.mean(z_values) if z_values != [] else depth[coord[0], coord[1]]
        return z_value

    
    def test_train(self):
        res = []
        if self.data_test is not None:
            for frame in self.data_test:
                try:
                    skel = frame['objects'][0]['skeleton']
                    gt_head = skel['head']['real']
                    gt_lw = skel['left_wrist']['real']
                    gt_rw = skel['right_wrist']['real']
                except KeyError:
                    continue
                predicted_coord = self.process((frame['depth'], frame['segment_id'], frame['image']))
                dh = min(np.linalg.norm(predicted_coord[0] - gt_head), 100)
                dlw = min(np.linalg.norm(predicted_coord[1] - gt_lw), 100)
                drw = min(np.linalg.norm(predicted_coord[2] - gt_rw), 100)
                res.append([dh, dlw, drw])
        return np.mean(res, axis=0)
    