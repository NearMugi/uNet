#!/usr/bin/env python
# coding: utf-8

# # KerasでU-Net  
# [U-NetでPascal VOC 2012の画像をSemantic Segmentationする (TensorFlow)](https://qiita.com/tktktks10/items/0f551aea27d2f62ef708)
# Kerasバージョンに変更する
# 
# * keras == 2.0.4  
# * tensorflow == 1.15.0  

# ## モデル

# In[1]:


import tensorflow as tf
from util import loader as ld


class UNet:
    def __init__(self, size=(128, 128), l2_reg=None):
        self.model = self.create_model(size, l2_reg)

    @staticmethod
    def create_model(size, l2_reg):
        inputs = tf.placeholder(tf.float32, [None, size[0], size[1], 3])
        teacher = tf.placeholder(tf.float32, [None, size[0], size[1], len(ld.DataSet.CATEGORY)])
        is_training = tf.placeholder(tf.bool)

        # 1, 1, 3
        conv1_1 = UNet.conv(inputs, filters=64, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        conv1_2 = UNet.conv(conv1_1, filters=64, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        pool1 = UNet.pool(conv1_2)

        # 1/2, 1/2, 64
        conv2_1 = UNet.conv(pool1, filters=128, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        conv2_2 = UNet.conv(conv2_1, filters=128, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        pool2 = UNet.pool(conv2_2)

        # 1/4, 1/4, 128
        conv3_1 = UNet.conv(pool2, filters=256, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        conv3_2 = UNet.conv(conv3_1, filters=256, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        pool3 = UNet.pool(conv3_2)

        # 1/8, 1/8, 256
        conv4_1 = UNet.conv(pool3, filters=512, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        conv4_2 = UNet.conv(conv4_1, filters=512, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        pool4 = UNet.pool(conv4_2)

        # 1/16, 1/16, 512
        conv5_1 = UNet.conv(pool4, filters=1024, l2_reg_scale=l2_reg)
        conv5_2 = UNet.conv(conv5_1, filters=1024, l2_reg_scale=l2_reg)
        concated1 = tf.concat([UNet.conv_transpose(conv5_2, filters=512, l2_reg_scale=l2_reg), conv4_2], axis=3)

        conv_up1_1 = UNet.conv(concated1, filters=512, l2_reg_scale=l2_reg)
        conv_up1_2 = UNet.conv(conv_up1_1, filters=512, l2_reg_scale=l2_reg)
        concated2 = tf.concat([UNet.conv_transpose(conv_up1_2, filters=256, l2_reg_scale=l2_reg), conv3_2], axis=3)

        conv_up2_1 = UNet.conv(concated2, filters=256, l2_reg_scale=l2_reg)
        conv_up2_2 = UNet.conv(conv_up2_1, filters=256, l2_reg_scale=l2_reg)
        concated3 = tf.concat([UNet.conv_transpose(conv_up2_2, filters=128, l2_reg_scale=l2_reg), conv2_2], axis=3)

        conv_up3_1 = UNet.conv(concated3, filters=128, l2_reg_scale=l2_reg)
        conv_up3_2 = UNet.conv(conv_up3_1, filters=128, l2_reg_scale=l2_reg)
        concated4 = tf.concat([UNet.conv_transpose(conv_up3_2, filters=64, l2_reg_scale=l2_reg), conv1_2], axis=3)

        conv_up4_1 = UNet.conv(concated4, filters=64, l2_reg_scale=l2_reg)
        conv_up4_2 = UNet.conv(conv_up4_1, filters=64, l2_reg_scale=l2_reg)
        outputs = UNet.conv(conv_up4_2, filters=ld.DataSet.length_category(), kernel_size=[1, 1], activation=None)

        return Model(inputs, outputs, teacher, is_training)

    @staticmethod
    def conv(inputs, filters, kernel_size=[3, 3], activation=tf.nn.relu, l2_reg_scale=None, batchnorm_istraining=None):
        if l2_reg_scale is None:
            regularizer = None
        else:
            regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg_scale)
        conved = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer
        )
        if batchnorm_istraining is not None:
            conved = UNet.bn(conved, batchnorm_istraining)

        return conved

    @staticmethod
    def bn(inputs, is_training):
        normalized = tf.layers.batch_normalization(
            inputs=inputs,
            axis=-1,
            momentum=0.9,
            epsilon=0.001,
            center=True,
            scale=True,
            training=is_training,
        )
        return normalized

    @staticmethod
    def pool(inputs):
        pooled = tf.layers.max_pooling2d(inputs=inputs, pool_size=[2, 2], strides=2)
        return pooled

    @staticmethod
    def conv_transpose(inputs, filters, l2_reg_scale=None):
        if l2_reg_scale is None:
            regularizer = None
        else:
            regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg_scale)
        conved = tf.layers.conv2d_transpose(
            inputs=inputs,
            filters=filters,
            strides=[2, 2],
            kernel_size=[2, 2],
            padding='same',
            activation=tf.nn.relu,
            kernel_regularizer=regularizer
        )
        return conved


class Model:
    def __init__(self, inputs, outputs, teacher, is_training):
        self.inputs = inputs
        self.outputs = outputs
        self.teacher = teacher
        self.is_training = is_training


# ## メイン処理

# In[ ]:


import argparse
import random
import tensorflow as tf

from util import loader as ld
from util import repoter as rp


def load_dataset(train_rate):
    loader = ld.Loader(
        dir_original="data_set/VOCdevkit/VOC2012/JPEGImages",
        dir_segmented="data_set/VOCdevkit/VOC2012/SegmentationClass",
    )
    return loader.load_train_test(train_rate=train_rate, shuffle=False)


def train(parser):
    # 訓練とテストデータを読み込みます
    # Load train and test datas
    train, test = load_dataset(train_rate=parser["trainrate"])
    valid = train.perm(0, 30)
    test = test.perm(0, 150)

    # 結果保存用のインスタンスを作成します
    # Create Reporter Object
    reporter = rp.Reporter(parser=parser)
    accuracy_fig = reporter.create_figure(
        "Accuracy", ("epoch", "accuracy"), ["train", "test"]
    )
    loss_fig = reporter.create_figure("Loss", ("epoch", "loss"), ["train", "test"])

    # GPUを使用するか
    # Whether or not using a GPU
    gpu = parser["gpu"]

    # モデルの生成
    # Create a model
    model_unet = UNet(l2_reg=parser["l2reg"]).model

    # 誤差関数とオプティマイザの設定をします
    # Set a loss function and an optimizer
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=model_unet.teacher, logits=model_unet.outputs
        )
    )
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    # 精度の算出をします
    # Calculate accuracy
    correct_prediction = tf.equal(
        tf.argmax(model_unet.outputs, 3), tf.argmax(model_unet.teacher, 3)
    )
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # セッションの初期化をします
    # Initialize session
    gpu_config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7),
        device_count={"GPU": 1},
        log_device_placement=False,
        allow_soft_placement=True,
    )
    sess = tf.InteractiveSession(config=gpu_config) if gpu else tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # 保存
    # saver = tf.train.Saver()

    # モデルの訓練
    # Train the model
    epochs = parser["epoch"]
    batch_size = parser["batchsize"]
    is_augment = parser["augmentation"]
    train_dict = {
        model_unet.inputs: valid.images_original,
        model_unet.teacher: valid.images_segmented,
        model_unet.is_training: False,
    }
    test_dict = {
        model_unet.inputs: test.images_original,
        model_unet.teacher: test.images_segmented,
        model_unet.is_training: False,
    }

    for epoch in range(epochs):
        for batch in train(batch_size=batch_size, augment=is_augment):
            # バッチデータの展開
            inputs = batch.images_original
            teacher = batch.images_segmented
            # Training
            sess.run(
                train_step,
                feed_dict={
                    model_unet.inputs: inputs,
                    model_unet.teacher: teacher,
                    model_unet.is_training: True,
                },
            )

        # 評価
        # Evaluation
        if epoch % 1 == 0:
            # saver.save(sess, "/ckpt/model.ckpt")
            loss_train = sess.run(cross_entropy, feed_dict=train_dict)
            loss_test = sess.run(cross_entropy, feed_dict=test_dict)
            accuracy_train = sess.run(accuracy, feed_dict=train_dict)
            accuracy_test = sess.run(accuracy, feed_dict=test_dict)
            print("Epoch:", epoch)
            print("[Train] Loss:", loss_train, " Accuracy:", accuracy_train)
            print("[Test]  Loss:", loss_test, "Accuracy:", accuracy_test)
            accuracy_fig.add([accuracy_train, accuracy_test], is_update=True)
            loss_fig.add([loss_train, loss_test], is_update=True)
            if epoch % 3 == 0:
                idx_train = random.randrange(10)
                idx_test = random.randrange(100)
                outputs_train = sess.run(
                    model_unet.outputs,
                    feed_dict={
                        model_unet.inputs: [train.images_original[idx_train]],
                        model_unet.is_training: False,
                    },
                )
                outputs_test = sess.run(
                    model_unet.outputs,
                    feed_dict={
                        model_unet.inputs: [test.images_original[idx_test]],
                        model_unet.is_training: False,
                    },
                )
                train_set = [
                    train.images_original[idx_train],
                    outputs_train[0],
                    train.images_segmented[idx_train],
                ]
                test_set = [
                    test.images_original[idx_test],
                    outputs_test[0],
                    test.images_segmented[idx_test],
                ]
                reporter.save_image_from_ndarray(
                    train_set,
                    test_set,
                    train.palette,
                    epoch,
                    index_void=len(ld.DataSet.CATEGORY) - 1,
                )

    # 訓練済みモデルの評価
    # Test the trained model
    loss_test = sess.run(cross_entropy, feed_dict=test_dict)
    accuracy_test = sess.run(accuracy, feed_dict=test_dict)
    print("Result")
    print("[Test]  Loss:", loss_test, "Accuracy:", accuracy_test)

    sess.close()

if __name__ == "__main__":
    parser = {
        "gpu": "store_true",
        "epoch" : 250,
        "batchsize" : 32,
        "trainrate" : 0.85,
        "augmentation" : "store_true",
        "l2reg" : 0.0001
    }
    print(parser)
    train(parser)


# In[ ]:




