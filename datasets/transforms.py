# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch
import torchvision.transforms
import torchvision.transforms.functional as F
import numpy as np
import random
from PIL import Image


class RandomHomography(torch.nn.Module):

    def __init__(self,  magnitude, directions='all'):
        super().__init__()
        self.magnitude = magnitude
        self.directions = directions.lower().replace(',', ' ').replace('-', ' ')

    def __repr__(self):
        return "RandomHomography(%g, '%s')" % (self.magnitude, self.directions)

    def homography_from_4pts(self, pts_cur, pts_new):
        "pts_cur and pts_new = 4x2 point array, in [(x,y),...] format"
        matrix = []
        for p1, p2 in zip(pts_new, pts_cur):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])
        A = np.matrix(matrix, dtype=np.float)
        B = np.array(pts_cur).reshape(8)

        homography = np.dot(np.linalg.pinv(A), B)
        homography = tuple(np.array(homography).reshape(8))
        return homography

    def forward(self, img):
        w, h = img.size

        x1, y1, x2, y2 = 0, 0, h, w
        original_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2)]

        max_skew_amount = max(w, h)
        max_skew_amount = int(np.ceil(max_skew_amount * self.magnitude))
        skew_amount = random.randint(1, max_skew_amount)

        if self.directions == 'all':
            choices = [0, 1, 2, 3]
        else:
            dirs = ['left', 'right', 'up', 'down']
            choices = []
            for d in self.directions.split():
                try:
                    choices.append(dirs.index(d))
                except:
                    raise ValueError('Tilting direction %s not recognized' % d)

        skew_direction =  random.randint(0, len(choices)-1)

        if skew_direction == 0:
            # Left Tilt
            new_plane = [(y1, x1 - skew_amount),  # Top Left
                         (y2, x1),  # Top Right
                         (y2, x2),  # Bottom Right
                         (y1, x2 + skew_amount)]  # Bottom Left
        elif skew_direction == 1:
            # Right Tilt
            new_plane = [(y1, x1),  # Top Left
                         (y2, x1 - skew_amount),  # Top Right
                         (y2, x2 + skew_amount),  # Bottom Right
                         (y1, x2)]  # Bottom Left
        elif skew_direction == 2:
            # Forward Tilt
            new_plane = [(y1 - skew_amount, x1),  # Top Left
                         (y2 + skew_amount, x1),  # Top Right
                         (y2, x2),  # Bottom Right
                         (y1, x2)]  # Bottom Left
        elif skew_direction == 3:
            # Backward Tilt
            new_plane = [(y1, x1),  # Top Left
                         (y2, x1),  # Top Right
                         (y2 + skew_amount, x2),  # Bottom Right
                         (y1 - skew_amount, x2)]  # Bottom Left

        # To calculate the coefficients required by PIL for the perspective skew,
        # see the following Stack Overflow discussion: https://goo.gl/sSgJdj
        homography = self.homography_from_4pts(original_plane, new_plane)
        img = img.transform(img.size, Image.PERSPECTIVE, homography, resample=Image.BICUBIC)
        return img




# "Pair": apply a transform on a pair
# "Both": apply the exact same transform to both images

class ComposePair(torchvision.transforms.Compose):
    def __call__(self, img1, img2):
        for t in self.transforms:
            img1, img2 = t(img1, img2)
        return img1, img2

class NormalizeBoth(torchvision.transforms.Normalize):
    def forward(self, img1, img2):
        img1 = super().forward(img1)
        img2 = super().forward(img2)
        return img1, img2

class ToTensorBoth(torchvision.transforms.ToTensor):
    def __call__(self, img1, img2):
        img1 = super().__call__(img1)
        img2 = super().__call__(img2)
        return img1, img2
        
class RandomRotationPair(torchvision.transforms.RandomRotation): 
    # the rotation will be intentionally different for the two images  with this class
    def forward(self, img1, img2):
        img1 = super().forward(img1)
        img2 = super().forward(img2)
        return img1, img2
        
class RandomHomographyPair(RandomHomography):
    # the homography will be intentionally different for the two images with this class
    def forward(self, img1, img2):
        img1 = super().forward(img1)
        img2 = super().forward(img2)
        return img1, img2
        
class RandomCropPair(torchvision.transforms.RandomCrop): 
    # the crop will be intentionally different for the two images with this class
    def forward(self, img1, img2):
        img1 = super().forward(img1)
        img2 = super().forward(img2)
        return img1, img2

class ColorJitterPair(torchvision.transforms.ColorJitter): 
    # can be symmetric (same for both images) or assymetric (different jitter params for each image) depending on assymetric_prob  
    def __init__(self, assymetric_prob, **kwargs):
        super().__init__(**kwargs)
        self.assymetric_prob = assymetric_prob
    def jitter_one(self, img, fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor):
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = F.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = F.adjust_hue(img, hue_factor)
        return img
        
    def forward(self, img1, img2):

        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )
        img1 = self.jitter_one(img1, fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor)
        if torch.rand(1) < self.assymetric_prob: # assymetric:
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue
            )
        img2 = self.jitter_one(img2, fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor)
        return img1, img2

def get_pair_transforms(transform_str, totensor=True, normalize=True):
    # transform_str is eg    crop224+color
    trfs = []
    for s in transform_str.split('+'):
        if s.startswith('crop'):
            size = int(s[len('crop'):])
            trfs.append(RandomCropPair(size))
        elif s=='acolor':
            trfs.append(ColorJitterPair(assymetric_prob=1.0, brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4), hue=0.0))
        elif s.startswith('rotate'):
            max_angle = float(s[len('rotate'):])
            trfs.append(RandomRotationPair(max_angle))
        elif s=='homography':
            trfs.append(RandomHomographyPair(magnitude=0.5))
        else:
            raise NotImplementedError('Unknown augmentation: '+s)
            
    if totensor:
        trfs.append( ToTensorBoth() )
    if normalize:
        trfs.append( NormalizeBoth(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) )

    if len(trfs)==0:
        return None
    elif len(trfs)==1:
        return trfs
    else:
        return ComposePair(trfs)
        
        
        
        
        
