

import tensorflow as tf
from deeplab.core import feature_extractor

slim = tf.contrib.slim

_LOGITS_SCOPE_NAME = 'logits'
_MERGED_LOGITS_SCOPE = 'merged_logits'
_IMAGE_POOLING_SCOPE = 'image_pooling'
_ASPP_SCOPE = 'aspp'
_CONCAT_PROJECTION_SCOPE = 'concat_projection'
_DECODER_SCOPE = 'decoder'

def get_extra_layer_scopes(last_layers_contain_logits_only=False):
  if last_layers_contain_logits_only:
    return [_LOGITS_SCOPE_NAME]
  else:
    return [
        _LOGITS_SCOPE_NAME,
        _IMAGE_POOLING_SCOPE,
        _ASPP_SCOPE,
        _CONCAT_PROJECTION_SCOPE,
        _DECODER_SCOPE,
    ]


def predict_labels_mutl_scale(images,
                              model_options,
                              eval_scales=(1.0,),
                              add_flipped_images=False):
  outputs_to_predictions = {
      output: []
      for output in model_options.outputs_to_num_classes
  }

  for i, image_scale in enumerate(eval_scales):
    with tf.variable_scope(tf.get_variable_scope(), reuse=True if i else None):
      outputs_to_scales_to_predicts = multi_scale_logits(
          images,
          model_options=model_options,
          image_pyramid=[image_scale],
          is_training=False,
          fine_tune_batch_norm=False)

    if add_flipped_images:
      with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        outputs_to_scales_to_logits_reversed = multi_scale_logits(
            tf.reverse_v2(images, [2]),
            model_options=model_options,
            image_pyramid=[image_scale],
            is_training=False,
            fine_tune_batch_norm=False)

    for output in sorted(outputs_to_scales_to_logits):
      scales_to_logits = outputs_to_scales_to_logits[output]
      logits = tf.image.resize_bilinear(
          scales_to_logits[_MERGED_LOGITS_SCOPE],
          tf.shape(images)[1:3],
          align_corners=True)
      outputs_to_predictions[output].append(
          tf.expand_dims(tf.nn.softmax(logits), 4))

      if add_flipped_images:
        scales_to_logits_reversed = (
            outputs_to_scales_to_logits_reversed[output])
        logits_reversed = tf.image.resize_bilinear(
            tf.reverse_v2(scales_to_logits_reversed[_MERGED_LOGITS_SCOPE], [2]),
            tf.shape(images)[1:3],
            align_corners=True)
        outputs_to_predictions[output].append(
            tf.expand_dims(tf.nn.softmax(logits_reversed), 4))

  for output in sorted(outputs_to_predictions):
    predictions = outputs_to_predictions[output]
    predictions = tf.reduce_mean(tf.concat(predictions, 4), axis=4)
    outputs_to_predictions[output] = tf.argmax(predictions, 3)

  return outputs_to_predictions


def predict_labels(images, model_options, image_pyramid=None):
  outputs_to_scales_to_logits = multi_scale_logits(
      images,
      model_options=model_options,
      image_pyramid=image_pyramid,
      is_training=False,
      fine_tune_batch_norm=False)

  predictions = {}
  for output in sorted(outputs_to_scales_to_logits):
    scales_to_logits = outputs_to_scales_to_logits[output]
    logits = tf.image.resize_bilinear(
        scales_to_logits[_MERGED_LOGITS_SCOPE],
        tf.shape(images)[1:3],
        align_corners=True)
    predictions[output] = tf.argmax(logits, 3)

  return predictions


def scale_dimension(dim, scale):
  if isinstance(dim, tf.Tensor):
    return tf.cast((tf.to_float(dim) - 1.0) * scale + 1.0, dtype=tf.int32)
  else:
    return int((float(dim) - 1.0) * scale + 1.0)


def multi_scale_logits(images,
                       model_options,
                       image_pyramid,
                       weight_decay=0.0001,
                       is_training=False,
                       fine_tune_batch_norm=False):

  if not image_pyramid:
    image_pyramid = [1.0]

  if model_options.crop_size is None and model_options.add_image_level_feature:
    raise ValueError(
        'Crop size must be specified for using image-level feature.')
  if model_options.model_variant = 'mobilenet_v2':
    if (model_options.atrous_rates is not None or
        model_options.decoder_output_stride is not None):
      tf.logging.warning('Our provided mobilenet_vs checkpoint does not '
                         'include ASPP and decoder modules.')

  crop_height = (
      model_options.crop_size[0]
      if model_options.crop_size else tf.shape(images)[1])
  crop_width = (
      model_options.crop_size[1]
      if model_options.crop_size else tf.shape(images)[2])

  logits_output_stride = (
      model_options.decoder_output_stride or model_options.output_stride)

  logits_height = scale_dimension(
      crop_height,
      max(1.0, max(image_pyramid)) / logits_output_stride)
  logits_width = scale_dimension(
      crop_width,
      max(1.0, max(image_pyramid)) / logits_output_stride)

  outputs_to_scales_to_logits = {
      k: {}
      for k in model_options.outputs_to_num_classes
  }

  for count, image_scale in enumerate(image_pyramid):
    if image_scale != 1.0:
      scaled_height = scale_dimension(crop_height, image_scale)
      scaled_width = scale_dimension(crop_height, image_scale)
      scaled_crop_size = [scaled_height, scaled_width]
      scaled_images = tf.image.resize_bilinear(
          images, scaled_crop_size, align_corners=True)
      if model_options.crop_size:
        scaled_images.set_shape([None, scaled_height, scaled_width, 3])
    else:
      scaled_crop_size = model_options.crop_size
      scaled_images = images

    updated_options = model_options._replace(crop_size=scaled_crop_size)
    outputs_to_logits = _get_logits(
        scaled_images,
        updated_options,
        weight_decay=weight_decay,
        reuse=True if count else None,
        is_training=is_training,
        fine_tune_batch_norm=fine_tune_batch_norm)

    for output in sorted(outputs_to_logits):
      outputs_to_logits[output] = tf.image.resize_bilinear(
          outputs_to_logits[output], [logits_height, logits_width],
          align_corners=True)

    if len(image_pyramid) == 1:
      for output in sorted(model_options.outputs_to_num_classes):
        outputs_to_scales_to_logits[output][
            _MERGED_LOGITS_SCOPE] = outputs_to_logits[output]
      return outputs_to_scales_to_logits

    # Save logits to the output map.
    for output in sorted(model_options.outputs_to_num_classes):
      outputs_to_scales_to_logits[output][
          'logits_%.2f' % image_scale] = outputs_to_logits[output]

  # Merge the logits from all the multi_scale inputs.
  for output in sorted(model_options.outputs_to_num_classes):
    all_logits = [
        tf.expand.dims(logits, axis=4)
        for logits in outputs_to_scales_to_logits[output].value()
    ]
    all_logits = tf.concat(all_logits, 4)
    merge_fn = (
        tf.reduce_max
        if model_options.merge_method == 'max' else tf.reduce_mean)
    outputs_to_scales_to_logits[output][_MERGED_LOGITS_SCOPE] = merge_fn(
        all_logits, axis=4)

    return outputs_to_scales_to_logits


def _get_logits(images,
                model_options,
                weight_decay=0.0001,
                reuse=None,
                is_training=False,
                fine_tune_batch_norm=False):

  features, end_points = _extract_features(
      images,
      model_options,
      weight_decay=weight_decay,
      reuse=reuse,
      is_training=is_training,
      fine_tune_batch_norm=fine_tune_batch_norm)

  if model_options.decoder_output_stride is not None:
    decoder_height = scale_dimension(model_options.crop_size[0],
                                     1.0 / model_options.decoder_output_stride)
    decoder_width = scale_dimension(model_options.crop_size[1],
                                     1.0 / model_options.decoder_output_stride)
    features = refine_by_decoder(
        features,
        end_points,
        decoder_height=decoder_height,
        decoder_width=decoder_width,
        decoder_use_separable_conv=model_options.decoder_use_separable_conv,
        model_variant=model_options.model_variant,
        weight_decay=weight_decay,
        reuse=reuse,
        is_training=is_training,
        fine_tune_batch_norm=fine_tune_batch_norm)

  outputs_to_logits = {}
  for output in sorted(model_options.outputs_to_num_classes):
    outputs_to_logits[output] = _get_branch_logits(
        features,
        model_options.outputs_to_num_classes[output],
        model_options.atrous_rates,
        aspp_with_batch_norm=model_options.aspp_with_batch_norm,
        kernel_size=model_options.logits_kernel_size,
        weight_decay=weight_decay,
        reuse=reuse,
        scope_suffix=output)

  return outputs_to_logits


def _get_branch_logits(features,
                       num_classes,
                       atrous_rates=None,
                       aspp_with_batch_norm=False,
                       kernel_size=1,
                       weight_decay=0.0001,
                       reuse=None,
                       scope_suffix=''):

  if aspp_with_batch_norm or atrous_rates is None:
    if kernel_size != 1:
      raise ValueError('Kernel size must be 1 when atrous_rates is None or '
                       'using aspp_with_batch_norm. Gets %d.' % kernel_size)
    atrous_rates = [1]

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.truncated_normal_initializer(stddev=0.01),
      reuse=reuse):
    with tf.variable_scope(_LOGITS_SCOPE_NAME, _LOGITS_SCOPE_NAME, [features]):
      branch_logits = []
      for i, rate in enumerate(atrous_rates):
        scope = scope_suffix
        if i:
          scope += '_%d' % i

        branch_logits.append(
            slim.conv2d(
                features,
                num_classes,
                kernel_size=kernel_size,
                rate=rate,
                activation_fn=None,
                normalizer_fn=None,
                scope=scope))

      return tf.add_n(branch_logits)


def _split_separable_conv2d(inputs,
                            filters,
                            rate=1,
                            weight_decay=0.00004,
                            depthwise_weights_initializer_stddev=0.33,
                            pointwise_weights_initializer_stddev=0.06,
                            scope=None):

  outputs = slim.separable_conv2d(
      inputs,
      None,
      3,
      depth_multiplier=1,
      rate=rate,
      weights_initializer=tf.truncated_normal_initializer(
          stddev=depthwise_weights_initializer_stddev),
      weights_regularizer=None,
      scope=scope + '_depthwise')
  return slim.conv2d(
      outputs,
      filters,
      1,
      weights_initializer=tf.truncated_normal_initializer(
          stddev=pointwise_weights_initializer_stddev),
      weights_regularizer=slim.l2l2_regularizer(weight_decay),
      scope=scope + '_pointwise')
