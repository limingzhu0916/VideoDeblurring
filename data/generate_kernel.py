import numpy as np
import cv2
import albumentations as albu
import tensorflow as tf
import torch

import cv2
import matplotlib.pyplot as plt
import numpy as np

def generate_kernel_trajectory(kernel_size, angle):
    """
    Generate blur kernel M, the blur kernel is symmetric and linear. M can be represent
    by a motion vector m=(l, o), the length of the motion vector l ∈ (0, kernel_size),
    the orientation of the motion vector o ∈ [0, angle)
    :param kernel_size: default 21, 31, 41
    :param angle: default 180
    :return: vector of motion
    """
    init_angle = np.random.uniform(0, angle)
    length = np.random.uniform(1, kernel_size)

def SubPixel_interpoaltion(kernel):
    """
    Applying sub-pixel interpolation to kernel trajectory
    :param kernel:
    :return:
    """
    return kernel

if __name__ == '__main__':
    """
    img = cv2.imread('E:/GoPro_Large/test/sharp/GOPR0384_11_05/004002.png')

    size = 15

    # generating the kernel
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size

    # applying the kernel to the input image
    output = cv2.filter2D(img, -1, kernel_motion_blur)

    cv2.imwrite('./blur.png', output)

    kernel = np.array([[0.25, 0.25, 0.25, 0.25], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    kernel = tf.constant(kernel, dtype=tf.float32)
    # image = Image.open('E:/GoPro_Large/test/sharp/GOPR0384_11_05/004002.png').convert('RGB')
    image = cv2.imread('E:/GoPro_Large/test/sharp/GOPR0384_11_05/004002.png')
    # image = transform(image)
    # h, w, *_ = image.shape
    # w = image.shape(2)
    # h = image.size(1)
    crop = albu.CenterCrop(256, 256, always_apply=True)
    normalize = albu.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    patch = crop(image=image)['image']
    cv2.imwrite('./sharp.png', patch)

    patch = normalize(image=patch)['image']
    # patch = np.transpose(patch, (2, 0, 1))
    _im = np.sign(patch) * (np.abs(patch)) ** 2.2
    # _im = torch.from_numpy(_im).unsqueeze(0)
    _im = tf.constant(_im, dtype=tf.float32)
    _im = tf.expand_dims(_im, 0)
    kernel = tf.expand_dims(kernel, 0)
    kernel = tf.expand_dims(kernel, 3)

    # _im = torch.sign(patch) * (torch.abs(patch)) ** 2.2
    # result = cv2.filter2D(_im, -1, kernel)
    # w_offset = random.randint(0, max(0, w - 256 - 1))
    # h_offset = random.randint(0, max(0, h - 256 - 1))

    # patch = image[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
    # patch = image[h_offset:h_offset + 256, w_offset:w_offset + 256]
    # _im = torch.sign(patch) * (torch.abs(patch)) ** 2.2
    c1 = tf.nn.conv2d(_im[:, :, :, 0:1], kernel, strides=[1, 1, 1, 1], padding='SAME')
    c2 = tf.nn.conv2d(_im[:, :, :, 1:2], kernel, strides=[1, 1, 1, 1], padding='SAME')
    c3 = tf.nn.conv2d(_im[:, :, :, 2:3], kernel, strides=[1, 1, 1, 1], padding='SAME')
    result = tf.concat([c1, c2, c3], axis=3)
    patch = tf.sign(result) * (tf.abs(result)) ** (1 / 2.2)
    patch_numpy = patch[0].float().numpy()
    patch_numpy = (np.transpose(patch_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    patch_numpy = patch_numpy.astype(np.uint8)
    cv2.imwrite('./blur.png', result)
    """
    kernel_size = 3
    sample = 10
    u0 = v0 = kernel_size / 2
    angle = 180
    for init_angle in range(0, angle):
        cos = np.cos(np.deg2rad(angle))
        if init_angle == 0:
            slope = 0
        else:
            slope = np.tan(np.deg2rad(angle))
        dx = lambda x: np.multiply(cos, x)
        trajectory_fun = lambda x: np.multiply(slope, x)
        for length in range(0, kernel_size + 1):
            if length == 0:
                kernel = complex(real=u0, imag=v0)
            else:
                max_half_length = length / 2
                iteration = sample * length + 1
                # kernel = np.array([complex(real=0, imag=0)] * iteration)
                kernel = np.array([complex(real=u0, imag=v0)])
                step = 0.1
                while step <= max_half_length:
                    du = dx(step)
                    u_right = u0 + du
                    v_right = v0 + trajectory_fun(u_right)
                    u_left = u0 - du
                    v_left = v0 + trajectory_fun(u_left)
                    kernel = np.append(kernel, np.array([complex(real=u_right, imag=v_right), complex(real=u_left, imag=v_left)]))
                    # kernel[int(step * sample) - 1] = complex(real=u_right, imag=v_right)
                    # kernel += complex(real=u_left, imag=v_left)
                    step += 0.1
                SubPixel_interpoaltion(kernel)
            plt.close()
            plt.plot(kernel.real, kernel.imag, '-', color='blue')

            plt.xlim((0, kernel_size))
            plt.ylim((0, kernel_size))
            plt.show()