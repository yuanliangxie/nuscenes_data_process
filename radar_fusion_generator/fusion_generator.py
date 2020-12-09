from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import box_in_image, view_points, BoxVisibility, points_in_box
from nuscenes.utils.data_classes import RadarPointCloud

from radar_fusion_generator.utils.fusion_projection import imageplus_creation, create_imagep_visualization
from radar_fusion_generator.utils.nuscenes_helper import get_sensor_sample_data, calc_mask
from radar_fusion_generator.utils import nuscence_dataset_preprocess, radar
from radar_fusion_generator.utils.noise_img import noisy

from train.params_init_nuScenes import Generate_Params as config

from typing import List, Tuple, Union
from pyquaternion.quaternion import Quaternion
from shapely.geometry import MultiPoint, box
from tqdm import tqdm
import numpy as np
import cv2
import math
import torch

class nuscenes_dataset(Dataset):
    def __init__(self, nusc, config, mode = "train"):
        """

        :param nusc: 数据集用来取数据的数据库
        :param config: 产生数据的参数
        :param mode: 因为有train和test，val分为两大类模式，所以数据增广不太一样！
        """
        self.nusc = nusc
        #self.normalize_bbox = False
        #self.only_radar_annotated = 0
        self.radar_sensors = config['radar_sensors']
        self.camera_sensors = config['camera_sensors']
        self.radar_cover_shape = config["radar_cover_shape"]#fusion_shape:"circle","line","normal_point"
        self.image_min_side = 450
        self.image_max_side = 800
        self.category_mapping = config["category_mapping"]
        self.classes, self.labels = self._get_class_label_mapping([c['name'] for c in nusc.category], self.category_mapping)
        #这里的_get_class_label_mapping可以改造成自己的class——maping二维的目标检测
        self.noisy_image_method = config["noisy_image_method"]
        self.noise_factor = config["noise_factor"]
        self.camera_dropout = config["camera_dropout"]
        self.radar_dropout = config["radar_dropout"]
        self.image_radar_fusion = config["image_radar_fusion"]
        #self.radar_projection_height = 2
        self.noise_filter = config["noise_filter"]
        self.n_sweeps = config["n_sweeps"]
        self.perfect_noise_filter = config["perfect_noise_filter"]
        self.visibilities = config["visibilities"]#设置bbox的可见度
        self.noise_category_selection = config["noise_category_selection"]#这里还要根据程序来变动,保留gt种类的radar，其余过滤
        self.normalize_radar = config["normalize_radar"]
        self.image_plus_creation = imageplus_creation
        #选择：                    rcs, vx, vy, dist, azimuth这5个参数
        self.channels = config["channels"]
        self.isdebug = config["isdebug"]
        self.image_plus_visual = config["image_plus_visual"]#是否在图片上显示radar的点或圆或线条
        self.total_transforms = nuscence_dataset_preprocess.Compose(transforms=[])
        if mode == "train":
            self.total_transforms.add(nuscence_dataset_preprocess.train_rvfusion_x_Augmentation())
        elif mode == "test":
            self.total_transforms.add(nuscence_dataset_preprocess.rvfusion_x_Augmentation())
        self.total_transforms.add(nuscence_dataset_preprocess.ToTensor(is_debug=self.isdebug))
        self.sample_tokens = {}
        f = open(config["sample_tokens_path"], 'r')
        for index, line in enumerate(f.readlines()):
            self.sample_tokens[index] = line.strip('\n')
        f.close()

    def load_sample_data(self, sample, sensor_channel, dtype = np.float32):
        """
        This function takes the token of a sample and a sensor sensor_channel and returns the according data

        Radar format: <np.array>
            - Shape: 18 x n
            - Semantics: x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0

        Image format: <np.array>
            - Shape: h x w x 3
            - Values: [0,255]
            - Channels: RGB
        """
        return get_sensor_sample_data(self.nusc, sample, sensor_channel, dtype=dtype, size=None)

    def load_image(self, image_index):
        """
        Returns the image plus from given image and radar samples.
        It takes the requested channels into account.

        :param sample_token: [str] the token pointing to a certain sample

        :returns: imageplus
        """
        # Initialize local variables
        radar_name = self.radar_sensors[0]
        camera_name = self.camera_sensors[0]

        # Gettign data from nuscenes database
        sample_token = self.sample_tokens[image_index]
        sample = self.nusc.get('sample', sample_token)

        # Grab the front camera and the radar sensor.
        radar_token = sample['data'][radar_name]
        camera_token = sample['data'][camera_name]
        image_target_shape = (self.image_min_side, self.image_max_side)

        # Load the image
        image_sample = self.load_sample_data(sample, camera_name, dtype=np.uint8)

        # Add noise to the image if enabled
        if self.noisy_image_method is not None and self.noise_factor > 0:
            image_sample = noisy(self.noisy_image_method, image_sample, self.noise_factor)

        if self.image_radar_fusion or self.camera_dropout > 0.0:

            # Parameters
            kwargs = {
                'pointsensor_token': radar_token,
                'camera_token': camera_token,
                'image_target_shape': image_target_shape,
                'clear_radar': np.random.rand() < self.radar_dropout,
                'clear_image': np.random.rand() < self.camera_dropout,
            }

            # Create image plus
            # radar_sample = self.load_sample_data(sample, radar_name) # Load samples from disk

            # Get filepath
            if self.noise_filter:#TODO:现在的noise_filter还没有实现，等待以后实现，所以self.noise_filter是置0的
                required_sweep_count = self.n_sweeps + self.noise_filter.num_sweeps_required - 1
            else:
                required_sweep_count = self.n_sweeps

            # sd_rec = self.nusc.get('sample_data', sample['data'][sensor_channel])
            sensor_channel = radar_name
            pcs, times = RadarPointCloud.from_file_multisweep(self.nusc, sample, sensor_channel, \
                                                              sensor_channel, nsweeps=required_sweep_count,
                                                              min_distance=0.0)

            # if self.noise_filter:
            #     # fill up with zero sweeps
            #     for _ in range(required_sweep_count - len(pcs)):
            #         pcs.insert(0, RadarPointCloud(np.zeros(shape=(RadarPointCloud.nbr_dims(), 0))))

            radar_sample = pcs.points #[radar.enrich_radar_data(pc.points) for pc in pcs]

            # if self.noise_filter:
            #     ##### Filter the pcs #####
            #     radar_sample = list(self.noise_filter.denoise(radar_sample, self.n_sweeps))
            #如果self.noise_filter这个函数做好了，则下面注释的可以释放
            # if len(radar_sample) == 0:
            #     radar_sample = np.zeros(shape=(len(radar.channel_map), 0))
            # else:
            #     ##### merge pcs into single radar samples array #####
            #     radar_sample = np.concatenate(radar_sample, axis=-1)

            radar_sample = radar_sample.astype(dtype=np.float32)
            radar_sample = radar.enrich_radar_data(radar_sample)#给多次扫描的radar增加测量数据

            if self.perfect_noise_filter:
                cartesian_uncertainty = 0.5  # meters
                angular_uncertainty = math.radians(1.7)  # degree
                category_selection = self.noise_category_selection

                nusc_sample_data = self.nusc.get('sample_data', radar_token)
                radar_gt_mask = calc_mask(nusc=self.nusc, nusc_sample_data=nusc_sample_data,
                                          points3d=radar_sample[0:3, :], \
                                          tolerance=cartesian_uncertainty, angle_tolerance=angular_uncertainty, \
                                          category_selection=category_selection)

                # radar_sample = radar_sample[:, radar_gt_mask.astype(np.bool)]
                radar_sample = np.compress(radar_gt_mask, radar_sample, axis=-1)

            if self.normalize_radar:
                # we need to noramlize
                # : use preprocess method analog to image preprocessing
                sigma_factor = int(self.normalize_radar)
                for ch in range(3, radar_sample.shape[0]):  # neural fusion requires x y and z to be not normalized
                    norm_interval = (0, 255)  # caffee mode is default and has these norm interval for img
                    radar_sample[ch, :] = radar.normalize(ch, radar_sample[ch, :], normalization_interval=norm_interval,
                                                          sigma_factor=sigma_factor)

            #print(radar_sample.shape)
            img_p_full, large_scale_p_full, middle_scale_p_full, little_scale_p_full = self.image_plus_creation(self.nusc, image_data=image_sample, radar_data=radar_sample, radar_cover_shape=self.radar_cover_shape,  **kwargs)

            img_p_full = img_p_full.astype(np.uint8)
            large_scale_p_full = large_scale_p_full.astype(np.uint8)
            middle_scale_p_full = middle_scale_p_full.astype(np.uint8)
            little_scale_p_full = little_scale_p_full.astype(np.uint8)

            #如果image_plus_visual等于true时，则进行渲染
            if self.image_plus_visual and self.isdebug: #在不debug时，程序不会进行渲染
            #######################img_p_full####################################
                input_data = self.create_input_data_plus_visual(img_p_full)
            #######################large_scale_p_full############################
                large_scale_input_data = self.create_input_data_plus_visual(large_scale_p_full)
            #######################middle_scale_p_full###########################
                middle_scale_input_data = self.create_input_data_plus_visual(middle_scale_p_full)
            #######################little_scale_p_full###########################
                little_scale_input_data = self.create_input_data_plus_visual(little_scale_p_full)

            else:
                input_data = img_p_full[:, :, self.channels]
                large_scale_input_data = large_scale_p_full[:, :, self.channels]
                middle_scale_input_data = middle_scale_p_full[:, :, self.channels]
                little_scale_input_data = little_scale_p_full[:, :, self.channels]

        else:  # We are not in image_plus mode
            # Only resize, because in the other case this is contained in image_plus_creation
            input_data = cv2.resize(image_sample, image_target_shape[::-1])

        return input_data, large_scale_input_data, middle_scale_input_data, little_scale_input_data

    def create_input_data_plus_visual(self, img_p_full):
        img_p_full[:, :, :3] = img_p_full[:, :, :3]
        image_pure_visual = create_imagep_visualization(img_p_full, draw_circles=False)
        img_p_full[:, :, :3] = image_pure_visual
        input_data = img_p_full[:, :, self.channels]
        return input_data


    def post_process_coords(self, corner_coords: List,
                            imsize: Tuple[int, int] = (1600, 900)) -> Union[Tuple[float, float, float, float], None]:
        """
        Get the intersection of the convex hull of the reprojected bbox corners and the image canvas, return None if no
        intersection.
        :param corner_coords: Corner coordinates of reprojected bounding box.
        :param imsize: Size of the image canvas.
        :return: Intersection of the convex hull of the 2D box corners and the image canvas.
        """
        polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
        img_canvas = box(0, 0, imsize[0], imsize[1])

        if polygon_from_2d_box.intersects(img_canvas):
            img_intersection = polygon_from_2d_box.intersection(img_canvas)
            intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])

            min_x = min(intersection_coords[:, 0])
            min_y = min(intersection_coords[:, 1])
            max_x = max(intersection_coords[:, 0])
            max_y = max(intersection_coords[:, 1])

            return min_x, min_y, max_x, max_y
        else:
            return None

    def get_2d_boxes(self, index: int, visibilities: List[str]) -> List[List]:
        """
        Get the 2D annotation records for a given `sample_data_token`.
        :param sample_data_token: Sample data token belonging to a camera keyframe.
        :param visibilities: Visibility filter.
        :return: [catagory, min_x, min_y, max_x, max_y],format:xyxy,absolute_coor_value
        List of 2D annotation record that belongs to the input `sample_data_token`

        """
        # Gettign data from nuscenes database
        sample_token = self.sample_tokens[index]
        sample = self.nusc.get('sample', sample_token)

        # Gettign sample_data_token from sensor['CAM_FRONT']
        sample_data_token = sample['data'][self.camera_sensors[0]]

        # Get the sample data and the sample corresponding to that sample data.
        sd_rec = self.nusc.get('sample_data', sample_data_token)

        assert sd_rec['sensor_modality'] == 'camera', 'Error: get_2d_boxes only works for camera sample_data!'
        if not sd_rec['is_key_frame']:
            raise ValueError('The 2D re-projections are available only for keyframes.')

        s_rec = self.nusc.get('sample', sd_rec['sample_token'])

        # Get the calibrated sensor and ego pose record to get the transformation matrices.
        cs_rec = self.nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        pose_rec = self.nusc.get('ego_pose', sd_rec['ego_pose_token'])
        camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

        # Get all the annotation with the specified visibilties.
        ann_recs = [self.nusc.get('sample_annotation', token) for token in s_rec['anns']]
        ann_recs = [ann_rec for ann_rec in ann_recs if (ann_rec['visibility_token'] in visibilities)]

        repro_recs = []
        bboxes = []

        for ann_rec in ann_recs:
            # Augment sample_annotation with token information.
            ann_rec['sample_annotation_token'] = ann_rec['token']
            ann_rec['sample_data_token'] = sample_data_token

            # Get the box in global coordinates.
            box = self.nusc.get_box(ann_rec['token'])

            # Move them to the ego-pose frame.
            box.translate(-np.array(pose_rec['translation']))
            box.rotate(Quaternion(pose_rec['rotation']).inverse)

            # Move them to the calibrated sensor frame.
            box.translate(-np.array(cs_rec['translation']))
            box.rotate(Quaternion(cs_rec['rotation']).inverse)

            # Filter out the corners that are not in front of the calibrated sensor.
            corners_3d = box.corners()
            in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
            corners_3d = corners_3d[:, in_front]

            # Project 3d box to 2d.
            corner_coords = view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist()

            # Keep only corners that fall within the image.
            final_coords = self.post_process_coords(corner_coords)

            # Skip if the convex hull of the re-projected corners does not intersect the image canvas.
            if final_coords is None:
                continue
            else:
                min_x, min_y, max_x, max_y = final_coords
                min_x = min_x/1600 * self.image_max_side
                max_x = max_x/1600 * self.image_max_side
                min_y = min_y/900 * self.image_min_side
                max_y = max_y/900 * self.image_min_side
                category = ann_rec['category_name']
            #这里进行了类别的过滤！
                if category in list(self.classes.keys()):
                    category_digit = self.classes[category]
                    bbox = [category_digit, min_x, min_y, max_x, max_y]
                    bboxes.append(bbox)
                else:
                    continue
        if bboxes == []:
            bboxes.append([-1, 0, 0, 0, 0])#添加-1为无anno,在可视化或者训练时要跳过处理
            print(sample_token)
            print("有一帧没有gt_box!!!")
            return bboxes
        return bboxes

    @staticmethod
    def _get_class_label_mapping(category_names, category_mapping):
        """
        :param category_mapping: [dict] Map from original name to target name. Subsets of names are supported.
            e.g. {'pedestrian' : 'pedestrian'} will map all pedestrian types to the same label

        :returns:
            [0]: [dict of (str, int)] mapping from category name to the corresponding index-number
            [1]: [dict of (int, str)] mapping from index number to category name
        """
        # Initialize local variables
        original_name_to_label = {}
        original_category_names = category_names.copy()
        #original_category_names.append('bg')
        if category_mapping is None:
            # Create identity mapping and ignore no class
            category_mapping = dict()
            for cat_name in category_names:
                category_mapping[cat_name] = cat_name

        # List of unique class_names
        selected_category_names = set(category_mapping.values())  # unordered
        selected_category_names = list(selected_category_names)
        selected_category_names.sort()  # ordered

        # Create the label to class_name mapping
        label_to_name = {label: name for label, name in enumerate(selected_category_names)}
        #label_to_name[len(label_to_name)] = 'bg'  # Add the background class

        # Create original class name to label mapping
        for label, label_name in label_to_name.items():

            # Looking for all the original names that are adressed by label name
            targets = [original_name for original_name in original_category_names if label_name in original_name]

            # Assigning the same label for all adressed targets
            for target in targets:
                # Check for ambiguity
                assert target not in original_name_to_label.keys(), 'ambigous mapping found for (%s->%s)' % (
                target, label_name)

                # Assign label to original name
                # Some label_names will have the same label, which is totally fine
                original_name_to_label[target] = label

        # Check for correctness
        actual_labels = original_name_to_label.values()
        expected_labels = range(0, max(actual_labels))  # we want to start labels at 0
        assert all([label in actual_labels for label in expected_labels]), 'Expected labels do not match actual labels'

        return original_name_to_label, label_to_name


    def __getitem__(self, index):
        labels = self.get_2d_boxes(index, visibilities=self.visibilities)
        image_plus, large_scale_image_plus, middle_scale_image_plus, little_scale_image_plus = self.load_image(index)
        large_scale_radar_image = large_scale_image_plus[:, :, 3:]
        middle_scale_radar_image = middle_scale_image_plus[:, :, 3:]
        little_scale_radar_image = little_scale_image_plus[:, :, 3:]
        image_sum_plus = np.concatenate([image_plus, large_scale_radar_image, middle_scale_radar_image, little_scale_radar_image], 2)
        #当训练时，要让self.image_plus_visual=Fasle
        labels = np.array(labels)

        # print(labels.shape)
        # print(index)
        label = labels[:, 0].reshape(-1, 1)
        boxes = labels[:, 1:]
        image, boxes, label = self.total_transforms(image_sum_plus, boxes, label)
        #self.channel只需要保留关键信息，由于image和image_plus固定，暂时不支持多尺度输入
        #需要有多部分输出，不仅需要输出image还需要输出radar_plus and image_sum_plus
        image_sum_plus = image
        # if self.isdebug:
        #     image_pure = image[:, :, :3]
        #     radar_plus = image[:, :, 3:]
        # else:
        #     image_pure = image[:3, :, :]
        #     radar_plus = image[3:, :, :]
        sample_token = self.sample_tokens[index]
        sample = {'sample_token': sample_token, 'image_sum_plus': image_sum_plus, 'label': torch.cat((label, boxes), 1)}
        return sample

    def __len__(self):
        return len(self.sample_tokens)

    def collate_fn(self, batch):
        '''Pad images and encode targets.

            As for images are of different sizes, we need to pad them to the same size.

            Args:
              batch: (list) of images, cls_targets, loc_targets.

            Returns:
              padded images, stacked cls_targets, stacked loc_targets.

            Reference:
              https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/utils/blob.py
            '''
        batchsize = len(batch)
        sample_tokens = [x['sample_token'] for x in batch]
        image_sum_plus = [x['image_sum_plus'] for x in batch]
        #radar_plus = [x['radar_plus'] for x in batch]
        #image_pure = [x['image_pure'] for x in batch]
        label = [x['label'] for x in batch]
        max_objects = max([ x.shape[0] for x in label])
        label_new = []
        for i in range(batchsize):
            label_len = label[i].shape[0]
            filled_labels = torch.zeros(max_objects, 5).float()
            filled_labels[range(max_objects)[:label_len]] = label[i][:]
            label_new.append(filled_labels)
        # filled_labels = np.zeros((max_objects, 1), dtype=np.float32)
        # filled_boxes = np.zeros((self.max_objects, 4), dtype=np.float32)
        # filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        #filled_boxes[range(len(boxes))[:self.max_objects]] = boxes[:self.max_objects]
        sample = {'sample_tokens': sample_tokens, 'image_sum_plus': torch.stack(image_sum_plus), 'label': torch.stack(label_new)}
        return sample

if __name__ == '__main__':#for debug
    nusc = NuScenes(version='v1.0-mini', dataroot='/home/xyl/PycharmProjects/crfnet-comback/v1.0-mini', verbose=True)
    # nusc = NuScenes(version='v1.0-trainval',
    #                 dataroot='/media/xyl/6418a039-786d-4cd8-b0bb-1ed36a649668/Datasets/nusecnes/train/v1.0-trainval01',
    #                 verbose=True)

    nudataset = nuscenes_dataset(nusc, config, mode="test")
    print(nudataset.classes)
    print(nudataset.labels)
    dataloader = torch.utils.data.DataLoader(nudataset, batch_size=1,
                                             shuffle=False, num_workers=0, pin_memory=False, collate_fn=nudataset.collate_fn)

    # from radar_fusion_generator.utils.video_save import video_save_from_capture
    # video_save = video_save_from_capture(video_name='video_only_annobox.avi', video_imgsize=(800, 450))
    for step, sample in enumerate(dataloader):
        for i, (image_sum_plus, label) in enumerate(zip(sample['image_sum_plus'], sample['label'])):
            image_sum_plus = image_sum_plus.numpy()
            image_pure = image_sum_plus[175:625, 0:800, :3]
            radar_plus_image = image_sum_plus[175:625, 0:800, 3:]
            all_radar_plus_image = radar_plus_image[:, :, :3]
            large_radar_plus_image = radar_plus_image[:, :, 5:8]
            middle_radar_plus_image = radar_plus_image[:, :, 10:13]
            little_radar_plus_image = radar_plus_image[:, :, 15:18]
            radar_plus_image = all_radar_plus_image
            label = label.numpy()
            for l in label:
                if l[0] == -1:#过滤掉不是类别的记号，因为某些图片中没有标记，但又没有去除掉这个图片
                    continue
                x1 = int((l[1] - l[3] / 2) * 800)
                y1 = int((l[2] - l[4] / 2) * 800)-175
                x2 = int((l[1] + l[3] / 2) * 800)
                y2 = int((l[2] + l[4] / 2) * 800)-175
                #image_pure = cv2.rectangle(image_pure, (x1, y1), (x2, y2), (255, 0, 255), thickness=2)
                #visualize_boxes(image=image_pure, boxes=np.array([[x1,y1,x2,y2]]), labels=np.array([int(l[0])]), probs= np.array([1]), class_labels=nudataset.classes)

                #cv2.putText(image_pure, nudataset.labels[l[0]], (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                #cv2.rectangle(radar_plus_image, (x1, y1), (x2, y2), (255, 0, 255), thickness=2)

            image_pure = cv2.cvtColor(image_pure, cv2.COLOR_RGB2BGR)
            radar_plus_image = cv2.cvtColor(radar_plus_image, cv2.COLOR_RGB2BGR)
            large_radar_plus_image = cv2.cvtColor(large_radar_plus_image, cv2.COLOR_RGB2BGR)
            middle_radar_plus_image = cv2.cvtColor(middle_radar_plus_image, cv2.COLOR_RGB2BGR)
            little_radar_plus_image = cv2.cvtColor(little_radar_plus_image, cv2.COLOR_RGB2BGR)
            ## cv2.imwrite("step{}_{}.jpg".format(step, i), image)
            #video_save.write(image)

            cv2.imshow('image_pure', image_pure)
            cv2.imshow('all_radar_plus_image', radar_plus_image)
            cv2.imshow('large_radar_plus_image', large_radar_plus_image)
            cv2.imshow('middle_radar_plus_image', middle_radar_plus_image)
            cv2.imshow('little_radar_plus_image', little_radar_plus_image)
            # cv2.imwrite("saved_img_{}_{}.png".format(step, config["radar_cover_shape"]), image_pure)
            # cv2.imwrite("saved_radar_img_{}_{}.png".format(step, config["radar_cover_shape"]), radar_plus_image)
            # cv2.waitKey(100)
            key = cv2.waitKey(0)
            if key == ord('s'):
                print("saving image")
                cv2.imwrite("saved_img_{}_{}.png".format(step, config["radar_cover_shape"]), image_pure)
                cv2.imwrite("saved_radar_img_{}_{}.png".format(step, config["radar_cover_shape"]), radar_plus_image)
            elif key == ord('p'):
                print("jump image!")
            elif key == ord('q'):
                print("exit!")
                break

    print("bye!")




