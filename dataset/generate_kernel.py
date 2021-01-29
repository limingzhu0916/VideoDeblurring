import matplotlib.pyplot as plt
import os
import numpy as np
from util import util

def __plot_kernels(kernel, show, save, path_to_save):
    """
    :param PSFs: kernel
    :param show: Whether to show kernel
    :param save: Whether to save kernel
    :param path_to_save: the path to save kernel
    """
    if len(kernel) == 0:
        raise Exception("Please run fit() method first.")
    else:
        plt.close()
        fig, axes = plt.subplots(1, 1, figsize=(10, 10))
        # fig, axes = plt.subplots(1, PSFnumber, figsize=(10, 10))
        # plot multi-PSF
        # for i in range(PSFnumber):
            # axes[i].imshow(kernel[i], cmap='gray')
        axes.imshow(kernel, cmap='gray')
        if show and save:
            if path_to_save is None:
                raise Exception('Please create Trajectory instance with path_to_save')
            plt.savefig(path_to_save)
            plt.show()
        elif save:
            if path_to_save is None:
                raise Exception('Please create Trajectory instance with path_to_save')
            plt.savefig(path_to_save)
        elif show:
            plt.show()

def generate_all_kernel_trajectory(kernel_size, angle, path_to_save, show=False, save=False):
    """
    Generate all blur kernel M, the blur kernel is symmetric and linear. M can be represent
    by a motion vector m=(l, o), the length of the motion vector l ∈ (0, kernel_size),
    the orientation of the motion vector o ∈ [0, angle), save all the kernel to disk
    :param kernel_size: default 21, 31, 41
    :param angle: default 180
    :param show: Whether to show kernel, default False
    :param save: Whether to save kernel, default False
    :param path_to_save: the path to save all kernels
    :return: vector of motion applying sub-pixel interpolation
    """
    u0 = v0 = kernel_size / 2
    dic = {}
    for init_angle in range(0, angle):
        cos = np.cos(np.deg2rad(init_angle))
        step = 0.5
        iteration = 0
        if init_angle == 0:
            slope = 0
        else:
            slope = np.tan(np.deg2rad(init_angle))
        dx = lambda x: np.multiply(cos, x)               # dx = cos(x)
        trajectory_fun = lambda x: np.multiply(slope, x)   # dy = slope * dx
        kernel = np.array([complex(real=u0, imag=v0)])     # The starting point is the center point
        for length in range(1, kernel_size-2):
            iterations = int(length)
            while iteration < iterations:
                du = dx(step)
                u_right = u0 + du
                v_right = v0 + trajectory_fun(du)
                u_left = u0 - du
                v_left = v0 - trajectory_fun(du)
                kernel = np.append(kernel, np.array([complex(real=u_right, imag=v_right), complex(real=u_left, imag=v_left)]))
                step += 0.5
                iteration += 1
            sub_kernel = SubPixel_interpoaltion(kernel_size, kernel)
            kernel_name = 'angle_%s_length_%s' % (init_angle, length)
            dic.setdefault(kernel_name, sub_kernel)

            # __plot_kernels(sub_kernel, show=show, save=save, path_to_save=path_to_save)
    util.mkdirs(path_to_save)
    np.save(os.path.join(path_to_save, 'kernel.npy'), dic)

def generate_kernel_trajectory(kernel_size, init_angle, length):
    u0 = v0 = kernel_size / 2
    step = 0.5
    iteration = 0
    cos = np.cos(np.deg2rad(init_angle))
    if init_angle == 0:
        slope = 0
    else:
        slope = np.tan(np.deg2rad(init_angle))
    dx = lambda x: np.multiply(cos, x)  # dx = cos(x)
    trajectory_fun = lambda x: np.multiply(slope, x)  # dy = slope * dx
    kernel = np.array([complex(real=u0, imag=v0)])  # The starting point is the center point
    while iteration < int(length):
        du = dx(step)
        u_right = u0 + du
        v_right = v0 + trajectory_fun(du)
        u_left = u0 - du
        v_left = v0 - trajectory_fun(du)
        kernel = np.append(kernel, np.array([complex(real=u_right, imag=v_right), complex(real=u_left, imag=v_left)]))
        step += 0.5
        iteration += 1
    sub_kernel = SubPixel_interpoaltion(kernel_size, kernel)
    return sub_kernel

def SubPixel_interpoaltion(kernel_size, kernel):
    """
    Applying sub-pixel interpolation to kernel trajectory
    """
    sub_kernel = np.zeros((kernel_size, kernel_size))
    triangle_fun = lambda x: np.maximum(0, (1 - np.abs(x)))
    triangle_fun_prod = lambda x, y: np.multiply(triangle_fun(x), triangle_fun(y))
    iterations = len(kernel)
    for t in range(iterations):

        m2 = int(np.minimum(kernel_size - 1, np.maximum(1, np.math.floor(kernel[t].real))))
        M2 = int(m2 + 1)
        m1 = int(np.minimum(kernel_size - 1, np.maximum(1, np.math.floor(kernel[t].imag))))
        M1 = int(m1 + 1)

        sub_kernel[m1, m2] += triangle_fun_prod(
            kernel[t].real - m2, kernel[t].imag - m1
        )
        sub_kernel[m1, M2] += triangle_fun_prod(
            kernel[t].real - M2, kernel[t].imag - m1
        )
        sub_kernel[M1, m2] += triangle_fun_prod(
            kernel[t].real - m2, kernel[t].imag - M1
        )
        sub_kernel[M1, M2] += triangle_fun_prod(
            kernel[t].real - M2, kernel[t].imag - M1
        )

    sub_kernel = sub_kernel / iterations
    return sub_kernel

if __name__ == '__main__':
    kernel_size = 31
    angle = 180
    generate_all_kernel_trajectory(kernel_size, angle, path_to_save='./kernels_31', show=False, save=False)
