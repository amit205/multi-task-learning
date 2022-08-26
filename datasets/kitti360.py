import numpy as np
import tensorflow as tf
from pathlib import Path
from glob import glob
import os
from .base_dataset import BaseDataset
from .utils import pipeline
#from superpoint.settings import DATA_PATH, EXPER_PATH
from utils.tools import dict_update
DATA_PATH = '/root/Internship-Valeo/Project/data/'
EXPER_PATH = '/root/Internship-Valeo/Project/data/KITTI-360'
import sys, getopt
import json


default_config = {
        'labels': None,
        'segmasks': None,
        'cache_in_memory': False,
        'validation_size': 100,
        'truncate': None,
        'preprocessing': {
            'resize': [240, 320]
        },
        'num_parallel_calls': 10,
        'augmentation': {
            'photometric': {
                'enable': False,
                'primitives': 'all',
                'params': {},
                'random_order': True,
            },
            'homographic': {
                'enable': False,
                'params': {},
                'valid_border_margin': 0,
            },
        },
        'warped_pair': {
            'enable': False,
            'params': {},
            'valid_border_margin': 0,
        },
    }
class Kitti(BaseDataset):
    def _init_dataset(self, **config):
        config = dict_update(default_config, config)
        # cat_2014 = '/root/Internship-Valeo/data/COCO/annotations_trainval2014/annotations/instances_val2014.json'
        # categories = []
        # json_file = cat_2014
        # if json_file is not None:
        #     with open(json_file, 'r') as COCO:
        #         js = json.loads(COCO.read())
        #         for i in range(80):
        #             categories.append(json.dumps(js['categories'][i]['name']).strip('""'))

        base_path = Path(DATA_PATH, 'KITTI-360' ,'train_images')
        #base_path = glob(DATA_PATH+'data_2d_raw/*/image_00/data_rect/*')
#        mask_path = Path(DATA_PATH, 'COCO/masktrain2014/')
        # a = []
        # x1 = []
        # x2 = []
        # x3 = []
        # for category in categories:
        #     a = os.listdir(os.path.join(base_path, '%s' % category))
        #     x1.extend((str(base_path)+'/' + category + '/' + s) for s in a)
        #     x2.extend((str(mask_path)+'/' + category + '/' + s) for s in a)
        #     x3.extend([category + '/' + s for s in a])
        # image_paths = iter(x1)
        # mask_paths = iter(x2)
        image_paths = list(base_path.iterdir())
        #image_paths = list(base_path)
        # mask_paths = list(mask_path.iterdir())
        if config['truncate']:
            image_paths = image_paths[:config['truncate']]
            # mask_paths = mask_paths[:config['truncate']]
        #names = [p[76:79]+p[104:114] for p in image_paths]
        names = [p.stem for p in image_paths]
        # names = x3
        image_paths = [str(p) for p in image_paths]
#        mask_paths = [str(Path(mask_path, name))+'.jpg' for name in names]
        # mask_paths = [str(p) for p in mask_paths]
        files = {'image_paths': image_paths, #'mask_paths': mask_paths, 
                 'names': names}
        if config['labels']:
            label_paths = []
            for n in names:
                #p = Path(EXPER_PATH, config['labels'],'{}.npz'.format("b'"+n+"'"))
                p = Path(EXPER_PATH, config['labels'],'{}.npz'.format(n))
                assert p.exists(), 'Image {} has no corresponding label {}'.format(n, p)
                label_paths.append(str(p))
            files['label_paths'] = label_paths
        if config['segmasks']:
            mask_paths = []
            for n in names:
                p = Path(EXPER_PATH, config['segmasks'],'{}.jpg'.format(n))
                assert p.exists(), 'Image {} has no corresponding segmentation_mask {}'.format(n, p)
                mask_paths.append(str(p))
            files['mask_paths'] = mask_paths
        tf.data.Dataset.map_parallel = lambda self, fn: self.map(
                fn, num_parallel_calls=config['num_parallel_calls'])

        return files

    def _get_data(self, files, split_name, **config):
        config = dict_update(default_config, config)
        has_keypoints = 'label_paths' in files
        has_seg_masks = 'mask_paths' in files
        is_training = split_name == 'training'

        def _read_image(path):
            image = tf.io.read_file(path)
            image = tf.image.decode_png(image, channels=3)
            return tf.cast(image, tf.float32)

        def _read_mask(path):
            image = tf.io.read_file(path)
            image = tf.image.decode_png(image)
            return tf.cast(image, tf.float32)

        def _preprocess(image):
            image = tf.image.rgb_to_grayscale(image)
            if config['preprocessing']['resize']:
                image = pipeline.ratio_preserving_resize(image,
                                                         **config['preprocessing'])
            return image
        def _preprocess_mask(mask_image):
            if config['preprocessing']['resize']:
                image = pipeline.ratio_preserving_mask_resize(mask_image,
                                                         **config['preprocessing'])
            return image

        # Python function
        def _read_points(filename):
            return np.load(filename.decode('utf-8'))['points'].astype(np.float32)

        names = tf.data.Dataset.from_tensor_slices(files['names'])
        images = tf.data.Dataset.from_tensor_slices(files['image_paths']) 
        images = images.map(_read_image)
        images = images.map(_preprocess)            
        data = tf.data.Dataset.zip({'image': images, #'mask_image': mask_images, 
                                    'name': names})

        # Add keypoints
        if has_keypoints:
            kp = tf.data.Dataset.from_tensor_slices(files['label_paths'])
            kp = kp.map(lambda path: tf.numpy_function(_read_points, [path], tf.float32))
            kp = kp.map(lambda points: tf.reshape(points, [-1, 2]))
            data = tf.data.Dataset.zip((data, kp)).map(
                    lambda d, k: {**d, 'keypoints': k})
            data = data.map(pipeline.add_dummy_valid_mask)
        if has_seg_masks:
            mask_images = tf.data.Dataset.from_tensor_slices(files['mask_paths'])
            mask_images = mask_images.map(_read_mask)
            mask_images = mask_images.map(_preprocess_mask)
            data = tf.data.Dataset.zip((data, mask_images)).map(
                    lambda d, k: {**d, 'mask_image': k})
        # Keep only the first elements for validation
        if split_name == 'validation':
            data = data.take(config['validation_size'])

        # Cache to avoid always reading from disk
        if config['cache_in_memory']:
            tf.logging.info('Caching data, fist access will take some time.')
            data = data.cache()

        # Generate the warped pair
        if config['warped_pair']['enable']:
            assert has_keypoints
            warped = data.map_parallel(lambda d: pipeline.homographic_augmentation(
                d, add_homography=True, **config['warped_pair']))
            if is_training and config['augmentation']['photometric']['enable']:
                warped = warped.map_parallel(lambda d: pipeline.photometric_augmentation(
                    d, **config['augmentation']['photometric']))
            warped = warped.map_parallel(pipeline.add_keypoint_map)
            # Merge with the original data
            data = tf.data.Dataset.zip((data, warped))
            data = data.map(lambda d, w: {**d, 'warped': w})
                                   # Data augmentation
        if has_keypoints and is_training:
            if config['augmentation']['photometric']['enable']:
                data = data.map_parallel(lambda d: pipeline.photometric_augmentation(
                    d, **config['augmentation']['photometric']))
            if config['augmentation']['homographic']['enable']:
                assert not config['warped_pair']['enable']  # doesn't support hom. aug.
                data = data.map_parallel(lambda d: pipeline.homographic_augmentation(
                    d, **config['augmentation']['homographic']))

        # Generate the keypoint map
        if has_keypoints:
            data = data.map_parallel(pipeline.add_keypoint_map)
        data = data.map_parallel(
            lambda d: {**d, 'image': tf.cast(d['image'], tf.float32) / 255.})

        if has_seg_masks:
            data = data.map_parallel(
            lambda d: {**d, 'mask_image': tf.cast(d['mask_image'], tf.float32) / 255})

        if config['warped_pair']['enable']:
            data = data.map_parallel(
                lambda d: {
                    **d, 'warped': {**d['warped'],
                                    'image': tf.cast(d['warped']['image'], tf.float32) / 255.}})
            if has_seg_masks:
                data = data.map_parallel(
                    lambda d: {
                        **d, 'warped': {**d['warped'],
                                        'mask_image': tf.cast(d['warped']['mask_image'], tf.float32) / 255}})

        return data
