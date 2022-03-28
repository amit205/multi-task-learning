import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers as tfl
from tensorflow.keras.layers import *

def vgg_block(inputs, filters, kernel_size, name, data_format, training=False,
              batch_normalization=True, kernel_reg=0., **params):
    x = tfl.Convolution2D(filters, kernel_size,
                       kernel_regularizer=tf.keras.regularizers.L2(kernel_reg),
                       data_format=data_format, **params)(inputs)
    if batch_normalization:
        x = tfl.BatchNormalization(
                    fused=True,
                    axis=1 if data_format == 'channels_first' else -1)(x)
    return x

def shared_encoder(shape, model_config):
    params_conv = {'padding': 'SAME', 'data_format': model_config['data_format'],
                   'batch_normalization': True, 'activation': tf.nn.relu,
                   'kernel_reg': model_config.get('kernel_reg', 0.)}
    params_pool = {'padding': 'SAME', 'data_format': model_config['data_format']}
    cfirst = model_config['data_format'] == 'channels_first'
    cindex = 1 if cfirst else -1  # index of the channel
    pool_size=(2, 2)
    kernel = 3
    inputs = Input(shape)
    # Encoder
    conv1 = vgg_block(inputs, 64, (kernel, kernel), 'conv1_1', **params_conv)
    conv2 = vgg_block(conv1, 64, (kernel, kernel), 'conv1_2', **params_conv)
    pool1 = MaxPooling2D(pool_size, name="block1_pool", **params_pool)(conv2)

    conv3 = vgg_block(pool1, 64, (kernel, kernel), 'conv2_1', **params_conv)
    conv4 = vgg_block(conv3, 64, (kernel, kernel), 'conv2_2', **params_conv)
    pool2 = MaxPooling2D(pool_size, name="block2_pool", **params_pool)(conv4)

    conv5 = vgg_block(pool2, 128, (kernel, kernel), 'conv3_1', **params_conv)
    conv6 = vgg_block(conv5, 128, (kernel, kernel), 'conv3_2', **params_conv)
    pool3 = MaxPooling2D(pool_size, name="block3_pool", **params_pool)(conv6)

    conv7 = vgg_block(pool3, 128, (kernel, kernel), 'conv4_1', **params_conv)
    conv8 = vgg_block(conv7, 128, (kernel, kernel), 'conv4_2', **params_conv)
    return keras.models.Model(inputs = inputs, outputs = conv8, name = 'shared_encoder')


def detector_head(shape, model_config):
    params_conv = {'padding': 'SAME', 'data_format': model_config['data_format'],
                   'batch_normalization': True,
                   'kernel_reg': model_config.get('kernel_reg', 0.)}
    cfirst = model_config['data_format'] == 'channels_first'
    cindex = 1 if cfirst else -1  # index of the channel

    inputs = Input(shape)
    x = vgg_block(inputs, 256, 3, 'conv1',
                      activation=tf.nn.relu, **params_conv)
    x = vgg_block(x, 1+pow(model_config['grid_size'], 2), 1, 'conv2',
                      activation=None, **params_conv)

    prob = tf.nn.softmax(x, axis=cindex)
    # Strip the extra “no interest point” dustbin
    prob = prob[:, :-1, :, :] if cfirst else prob[:, :, :, :-1]
    prob = tf.nn.depth_to_space(
              prob, model_config['grid_size'], data_format='NCHW' if cfirst else 'NHWC')
    prob = tf.squeeze(prob, axis=cindex)
#     return {'logits': x, 'prob': prob}
    return keras.models.Model(inputs = inputs, outputs = {'logits': x, 'prob': prob}, name = 'detector_head')


def descriptor_head(shape, model_config):
    params_conv = {'padding': 'SAME', 'data_format': model_config['data_format'],
                   'batch_normalization': True,
                   'kernel_reg': model_config.get('kernel_reg', 0.)}
    
    cfirst = model_config['data_format'] == 'channels_first'
    cindex = 1 if cfirst else -1  # index of the channel
    inputs = Input(shape)
    x = vgg_block(inputs, 256, 3, 'conv1',
                      activation=tf.nn.relu, **params_conv)
    x = vgg_block(x, model_config['descriptor_size'], 1, 'conv2',
                      activation=None, **params_conv)

    desc = tf.transpose(x, [0, 2, 3, 1]) if cfirst else x
    desc = tf.image.resize_bilinear(
            desc, model_config['grid_size'] * tf.shape(desc)[1:3])
    desc = tf.transpose(desc, [0, 3, 1, 2]) if cfirst else desc
    desc = tf.nn.l2_normalize(desc, cindex)

#    return {'descriptors_raw': x, 'descriptors': desc}
    return keras.models.Model(inputs = inputs, outputs = {'descriptors_raw': x, 'descriptors': desc}, name = 'descriptor_head')

def detector_loss(keypoint_map, logits, model_config, valid_mask=None):
    if model_config['data_format'] == 'channels_first':
        logits = tf.transpose(logits, [0, 2, 3, 1])
    # Convert the boolean labels to indices including the "no interest point" dustbin
    labels = keypoint_map[..., tf.newaxis]  # for GPU
    labels = tf.cast(labels, tf.float32)
    labels = tf.nn.space_to_depth(labels, model_config['grid_size'])
    shape = tf.concat([tf.shape(labels)[:3], [1]], axis=0)
    labels = tf.concat([2*labels, tf.ones(shape)], 3)
    # Add a small random matrix to randomly break ties in argmax
    labels = tf.argmax(labels + tf.random.uniform(tf.shape(labels), 0, 0.1), axis=3)
    # Mask the pixels if bordering artifacts appear
    valid_mask = tf.ones_like(keypoint_map) if valid_mask is None else valid_mask
    valid_mask = valid_mask[..., tf.newaxis]  # for GPU
    valid_mask = tf.cast(valid_mask, tf.float32)
    valid_mask = tf.nn.space_to_depth(valid_mask, model_config['grid_size'])
    valid_mask = tf.math.reduce_prod(valid_mask, axis=3)  # AND along the channel dim
    valid_mask = tf.cast(valid_mask, tf.int64)
#     labels = labels * valid_mask
#     loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, weights=valid_mask)
    return loss

def model_metrics(y_true, y_pred, valid_mask):
#     pred = tf.expand_dims(pred, axis = 3)
    pred = valid_mask * y_pred
    labels = y_true
    precision = tf.math.reduce_sum(pred * labels) / tf.math.reduce_sum(pred)
    recall = tf.math.reduce_sum(pred * labels) / tf.math.reduce_sum(labels)
    return {'precision': precision, 'recall': recall}


def box_nms(prob, size, iou=0.1, min_prob=0.01, keep_top_k=0):
    """Performs non maximum suppression on the heatmap by considering hypothetical
    bounding boxes centered at each pixel's location (e.g. corresponding to the receptive
    field). Optionally only keeps the top k detections.
    Arguments:
        prob: the probability heatmap, with shape `[H, W]`.
        size: a scalar, the size of the bouding boxes.
        iou: a scalar, the IoU overlap threshold.
        min_prob: a threshold under which all probabilities are discarded before NMS.
        keep_top_k: an integer, the number of top scores to keep.
    """
    with tf.name_scope('box_nms'):
        pts = tf.cast(tf.where(tf.greater_equal(prob, min_prob)), tf.float32)
        size = tf.constant(size/2.)
        boxes = tf.concat([pts-size, pts+size], axis=1)
        scores = tf.gather_nd(prob, tf.cast(pts, tf.int32))
        indices = tf.image.non_max_suppression(boxes, scores, tf.shape(boxes)[0], iou)
        pts = tf.gather(pts, indices)
        scores = tf.gather(scores, indices)
        if keep_top_k:
            k = tf.minimum(tf.shape(scores)[0], tf.constant(keep_top_k))  # when fewer
            scores, indices = tf.nn.top_k(scores, k)
            pts = tf.gather(pts, indices)
        prob = tf.scatter_nd(tf.cast(pts, tf.int32), scores, tf.shape(prob))
    return prob
