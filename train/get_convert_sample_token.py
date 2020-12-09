from nuscenes.nuscenes import NuScenes
from train.params_init_nuScenes import Generate_Params as config_gener
from nuscenes.utils.geometry_utils import view_points
from pyquaternion.quaternion import Quaternion
from shapely.geometry import MultiPoint, box
import numpy as np
import json
from tqdm import tqdm
import random
from random import randint
import os

class get_convert_sample_token(object):
    def __init__(self, nusc, config_gener):
        """
        本程序集成了将sample_token写入txt文件的程序和将sample_token中的groundtruth标注信息写入到json文件中，后面将输入到coco_eval去测评！
        :param nusc:
        :param config_gener:
        """
        self.camera_sensors = config_gener['camera_sensors']
        self.image_min_side = 450
        self.image_max_side = 800
        self.category_mapping = config_gener["category_mapping"]
        self.classes_2label, self.labels_2class = self._get_class_label_mapping([c['name'] for c in nusc.category], self.category_mapping)
        self.nusc = nusc
        self.visibilities = config_gener["visibilities"]
        self.train_scenes = []
        self.val_scenes = []
        self.test_scenes = []
        self.sample_tokens = []
        self.sample_tokens_train = []
        self.sample_tokens_val = []
        self.sample_tokens_test = []
        self.align_scenes_train_val_test()
        self.generate_tran_val_test()
        # self.find_all_keyfram_sample_token()
        # self.filter_with_no_gt_boxes()#已经过滤掉了没有gt_boxes的图片
        # self.align_train_val_test_tokens()
        self.START_BOUNDING_BOX_ID = 1

    def align_scenes_train_val_test(self):
        nusc = self.nusc
        scene_indices = range(len(nusc.scene))
        random.seed(1)
        for scene_index in scene_indices:
            rand_digit = randint(1, 10)
            if rand_digit <= 6:
                self.train_scenes.append(scene_index)
            elif rand_digit <= 8:
                self.val_scenes.append(scene_index)
            elif rand_digit <= 10:
                self.test_scenes.append(scene_index)



    def find_all_keyfram_sample_token(self, scene_indices):
        sample_tokens= []
        nusc = self.nusc
        num_samples = 0
        print("\nfind all the key frams in scenes\n")
        for scene_index in tqdm(scene_indices):
            first_sample_token = nusc.scene[scene_index]['first_sample_token']
            # nbr_samples = nusc.scene[scene_index]['nbr_samples']
            curr_sample = nusc.get('sample', first_sample_token)

            while (curr_sample):
                curr_sample_data_CAM_Sensor = nusc.get('sample_data', curr_sample['data'][self.camera_sensors[0]])
                if curr_sample_data_CAM_Sensor['is_key_frame'] == True:
                    sample_tokens.append(curr_sample['token'])
                    if curr_sample['next']:
                        next_token = curr_sample['next']
                        curr_sample = nusc.get('sample', next_token)
                        num_samples += 1
                    else:
                        num_samples += 1
                        break
                else:
                    pass  # 只有key_frame才能加入到sample_token！
        return sample_tokens

    def filter_with_no_gt_boxes(self, sample_tokens):
        """
        分配sample_token给train, val, test
        :return:
        """
        sample_tokens_filter = []
        print("\nstart to filt the sample token which has gt_boxes\n")
        for sample_token in sample_tokens:
            sample_token_index = sample_tokens.index(sample_token)

            #过滤掉不在文件夹的sample_token
            sample = nusc.get('sample', sample_token)
            sample_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
            filename = sample_data["filename"]
            file_path = os.path.join(self.nusc.dataroot, filename)
            if not os.path.isfile(file_path):
                print("pass no exist sample_token:{} now".format(sample_token_index))
                continue
            #过滤掉没有bboxes的sample_token
            bboxes = self.get_2d_boxes(sample_token)
            if bboxes==[]:
                print(sample_token_index, "!!!")
                continue
            sample_tokens_filter.append(sample_token)
        return sample_tokens_filter

    def generate_tran_val_test(self):
        ###################find_sample_tokens###################
        sample_tokens_train = self.find_all_keyfram_sample_token(self.train_scenes)
        sample_tokens_val = self.find_all_keyfram_sample_token(self.val_scenes)
        sample_tokens_test = self.find_all_keyfram_sample_token(self.test_scenes)
        ###################fileter__sample_tokens###############
        self.sample_tokens_train = self.filter_with_no_gt_boxes(sample_tokens_train)
        self.sample_tokens_val = self.filter_with_no_gt_boxes(sample_tokens_val)
        self.sample_tokens_test = self.filter_with_no_gt_boxes(sample_tokens_test)

    # def align_train_val_test_tokens(self):
    #     # 进行分类，进行train, val, test
    #     random.seed(1)
    #     for sample_token in self.sample_tokens:
    #         rand_digit = randint(1, 10)
    #         if rand_digit <= 6:
    #             self.sample_tokens_train.append(sample_token)
    #         elif rand_digit <= 8:
    #             self.sample_tokens_val.append(sample_token)
    #         elif rand_digit <= 10:
    #             self.sample_tokens_test.append(sample_token)


    def write_train_sample_token(self):
        f = open(config_gener["train_txt_path"], 'w')
        print("\nwrite valid sample_token into text\n")
        for i in tqdm(range(len(self.sample_tokens_train))):
            f.write(self.sample_tokens_train[i]+'\n')
        f.close()
        print("write_path:" + config_gener["train_txt_path"])

    def write_val_sample_token(self):
        f = open(config_gener["val_txt_path"], 'w')
        print("\nwrite valid sample_token into text\n")
        for i in tqdm(range(len(self.sample_tokens_val))):
            f.write(self.sample_tokens_val[i] + '\n')
        f.close()
        print("write_path:" + config_gener["val_txt_path"])

    def write_test_sample_token(self):
        f = open(config_gener["test_txt_path"], 'w')
        print("\nwrite valid sample_token into text\n")
        for i in tqdm(range(len(self.sample_tokens_test))):
            f.write(self.sample_tokens_test[i]+'\n')
        f.close()
        print("write_path:" + config_gener["test_txt_path"])



    def get_2d_boxes(self, sample_token):
        """
        Get the 2D annotation records for a given `sample_data_token`.
        :param sample_data_token: Sample data token belonging to a camera keyframe.
        :param visibilities: Visibility filter.
        :return: [catagory, min_x, min_y, max_x, max_y],format:xyxy,absolute_coor_value
        List of 2D annotation record that belongs to the input `sample_data_token`

        """
        # Gettign data from nuscenes database
        #sample_token = self.sample_tokens[index]
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
        ann_recs = [ann_rec for ann_rec in ann_recs if (ann_rec['visibility_token'] in self.visibilities)]

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
                if category in list(self.classes_2label.keys()):
                    category_digit = self.classes_2label[category]
                    bbox = [category_digit, min_x, min_y, max_x, max_y]
                    bboxes.append(bbox)
                else:
                    continue
        return bboxes
    def post_process_coords(self, corner_coords, imsize = (1600, 900)):
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

    def convert_to_json(self, json_file, sample_tokens):
        """

        :param json_file:json_file所要写文件的地址
        :param sample_tokens: 所要写json文件的测试或者验证sample_tokens列表：list
        :return:
        """
        json_dict = {"images": [], "type": "instances", "annotations": [],
                     "categories": []}
        categories = self.classes_2label
        bnd_id = self.START_BOUNDING_BOX_ID
        camera_name = "CAM_FRONT"#TODO：这里可以再弄循环
        print("Processing sample_tokens" )
        for sample_token in tqdm(sample_tokens):
            # print("Processing %s" % (sample_token))
            sample = nusc.get('sample', sample_token)
            sample_data = nusc.get("sample_data", sample["data"][camera_name])
            image_id = sample_data["filename"].split('/')[-1]  #
            filename = sample_data["filename"].split('/')[-1]  #这里将file_name 和 image_id 设置为文件名
            width = sample_data["width"]
            height = sample_data["height"]
            image = {'file_name': filename, 'height': height, 'width': width,
                     'id': image_id}
            json_dict['images'].append(image)
            ## Cruuently we do not support segmentation
            #  segmented = get_and_check(root, 'segmented', 1).text
            #  assert segmented == '0'

            bboxes=self.get_2d_boxes(sample_token)
            for bbox in bboxes:
                category_id, xmin, ymin, xmax, ymax = bbox#TODO:这里采用的坐标是（450， 800的坐标，所以测试输入的图片要是（450，800）尺寸的图片）
                assert (xmax > xmin)
                assert (ymax > ymin)
                o_width = abs(xmax - xmin)
                o_height = abs(ymax - ymin)
                ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id':
                    image_id, 'bbox': [xmin, ymin, o_width, o_height],
                       'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                       'segmentation': []}
                json_dict['annotations'].append(ann)
                bnd_id = bnd_id + 1

        for cate, cid in categories.items():
            cat = {'supercategory': 'none', 'id': cid, 'name': cate}
            json_dict['categories'].append(cat)
        json_fp = open(json_file, 'w')
        json_str = json.dumps(json_dict)
        json_fp.write(json_str)
        json_fp.close()


if __name__ == '__main__':
    nusc = NuScenes(version='v1.0-mini', dataroot='/home/xyl/PycharmProjects/crfnet-comback/v1.0-mini', verbose=True)
    #nusc = NuScenes(version='v1.0-trainval', dataroot='/media/xyl/6418a039-786d-4cd8-b0bb-1ed36a649668/Datasets/nusecnes/train/v1.0-trainval01', verbose=True)
    train_txt_path = "../data/nuscence/train.txt"
    val_txt_path = "../data/nuscence/val.txt"
    test_txt_path = "../data/nuscence/test.txt"
    config_gener["train_txt_path"] = train_txt_path
    config_gener["val_txt_path"] = val_txt_path
    config_gener["test_txt_path"] = test_txt_path

    path_data = "../data/nuscence"
    path_evaluate_coco = "../evaluate_coco"
    if not os.path.exists(path_data):
        os.makedirs(path_data)
    if not os.path.exists(path_evaluate_coco):
        os.makedirs(path_evaluate_coco)

    divide_sample_token_xie = get_convert_sample_token(nusc, config_gener)
    divide_sample_token_xie.write_train_sample_token()
    divide_sample_token_xie.write_val_sample_token()
    divide_sample_token_xie.write_test_sample_token()
    divide_sample_token_xie.convert_to_json('../evaluate_coco/test_gt_json_450x800.json', divide_sample_token_xie.sample_tokens_test)
    divide_sample_token_xie.convert_to_json("../evaluate_coco/val_gt_json_450x800.json", divide_sample_token_xie.sample_tokens_val)