import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
from imgaug import augmenters as iaa

class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):#TODO：这里出了问题！
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels

    def add(self, transform):
        self.transforms.append(transform)

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        x_c = boxes[:, 0] * width
        w = boxes[:, 2] * width
        y_c = boxes[:, 1] * height
        h = boxes[:, 3] * height
        #xywh tran_to x1y1x2y2
        # boxes[:, 0] = x_c - w / 2
        # boxes[:, 1] = y_c - h / 2
        # boxes[:, 2] = x_c + w / 2
        # boxes[:, 3] = y_c + h / 2
        boxes = np.stack((x_c, y_c, w, h), axis=1)
        return image, boxes, labels
class xywh_to_xyxy(object):
    def __call__(self, image, boxes=None, labels=None):
        x_c = boxes[:, 0].copy()#当不做运算时,x_c= boxes[:,0],出现问题，x_c只是boxes[:,0]的引用而已，容易出错，因为下面的代码出现了这样的错误
        y_c = boxes[:, 1].copy()
        w = boxes[:, 2]
        h = boxes[:, 3]

        boxes[:, 0] = x_c - w / 2
        boxes[:, 1] = y_c - h / 2
        boxes[:, 2] = x_c + w / 2
        boxes[:, 3] = y_c + h / 2
        return image, boxes, labels

class xyxy_to_xywh(object):
    def __call__(self, image, boxes=None, labels=None):
        x1 = boxes[:, 0].copy()
        y1 = boxes[:, 1].copy()
        x2 = boxes[:, 2].copy()
        y2 = boxes[:, 3].copy()
        boxes[:, 0] = (x1 + x2) / 2
        boxes[:, 1] = (y1 + y2) / 2
        boxes[:, 2] = x2 - x1
        boxes[:, 3] = y2 - y1
        return image, boxes, labels



class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class ResizeImage(object):
    def __init__(self, interpolation=cv2.INTER_AREA):
        self.interpolation = interpolation
        #self.image_size = {1: (320, 320), 2: (416, 416), 3: (608, 608)}
        self.new_size = (800, 800)

    def __call__(self, image, boxes=None, labels=None):
        image, boxes = self.cv2_letterbox_image(image, boxes, self.new_size)
        return image, boxes, labels

    def cv2_letterbox_image(self, image, boxes, expected_size):
        """
        使图片固定aspect ratio，并且更新boxes
        :param image:image_plus
        :param boxes: shape = [:,4], format:xyxy
        :param expected_size:(,)
        :return:
        """
        image_plus_part = image[:, :, 3:]
        image = image[:, :, :3]
        ih, iw = image.shape[0:2]
        ew, eh = expected_size
        new_image_plus_part = np.zeros((eh, ew, image_plus_part.shape[2]))
        scale = min(eh / ih, ew / iw)
        boxes = boxes * scale
        nh = int(ih * scale)
        nw = int(iw * scale)
        image = cv2.resize(image, (nw, nh), interpolation=self.interpolation)
        top = (eh - nh) // 2
        bottom = eh - nh - top
        left = (ew - nw) // 2
        right = ew - nw - left
        boxes[:, [0,2]] = boxes[:, [0,2]] + left
        boxes[:, [1,3]] = boxes[:, [1,3]] + top#boxes里的坐标不会出界，因为此图片扩充，框永远在图片中
        new_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        new_image_plus_part[top: top+ih, left:left+nw, :] = image_plus_part
        new_img = np.concatenate((new_img, new_image_plus_part),axis=2)
        return new_img, boxes


class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes

class ToTensor(object):
    def __init__(self, is_debug=False):
        #self.max_objects = max_objects
        self.is_debug = is_debug

    def __call__(self, image, boxes=None, labels=None):
        if self.is_debug == False:
            image = image.astype(np.float32)
            image /= 255.0
            image = np.transpose(image, (2, 0, 1))  # 把channel调到前面
            return torch.from_numpy(image).type(torch.FloatTensor), \
                   torch.from_numpy(boxes).type(torch.FloatTensor), torch.from_numpy(labels).type(torch.FloatTensor)

        else:
            image_plus = image.astype(np.uint8)#将image_plus全部转回np.uint8类型
            return torch.from_numpy(image_plus), torch.from_numpy(boxes).float(), torch.from_numpy(labels).float()
        # filled_labels = np.zeros((self.max_objects, 1), dtype=np.float32)
        # filled_boxes = np.zeros((self.max_objects, 4), dtype=np.float32)
        # filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        # filled_boxes[range(len(boxes))[:self.max_objects]] = boxes[:self.max_objects]
        #其中有些帧的没有label，但是为了连续性，也添加了lable，但是在类别上是-1,所以在训练时类别是-1的要跳过处理。(后话,这个问题已经被解决了！)

class ImageBaseAug(object):
    def __init__(self):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        self.seq = iaa.Sequential([
                              sometimes(iaa.Multiply((0.5, 1.5), per_channel=0.2)),
                              sometimes(iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5))],
                             random_order=True)


    def __call__(self, image, boxes, labels):

        image[:, :, :3] = self.seq(image=image[:, :, :3])
        return image, boxes, labels

class Expand(object):
    def __init__(self, mean=0):
        self.mean = mean
        """
        Expand函数将图片扩展，并使原始大小图片在扩展的图片中任意放置！
        """

    def __call__(self, image, boxes, labels):
        # if random.randint(2):#有一定几率返回原图！
        #     return image, boxes, labels
        height, width, depth = image.shape
        #ratio = random.uniform(1, 4)
        width_set = 800
        height_set = 800
        left = random.uniform(0, width_set - width)
        top = random.uniform(0, height_set - height)

        expand_image = np.zeros(
            (800, 800, depth),
            dtype=image.dtype)
        mean = np.array([self.mean]*depth, dtype=image.dtype)
        #mean = np.array([self.mean]).astype(np.uint8)
        expand_image[:, :, :] = mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image
        boxes = boxes.copy()
        boxes[:, :2] += (left, top)
        boxes[:, 2:] += (left, top)

        return image, boxes, labels

class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, depth = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels



class rvfusion_x_Augmentation(object):
    def __init__(self):
        self.augment = Compose([
            ResizeImage(),
            xyxy_to_xywh(),
            ToPercentCoords(),
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)

class train_rvfusion_x_Augmentation(object):
    def __init__(self):
        self.augment = Compose([
            RandomSampleCrop(),
            ImageBaseAug(),
            Expand(0),
            RandomMirror(),
            ResizeImage(),
            xyxy_to_xywh(),
            ToPercentCoords(),
        ])
    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)