from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time as tm
import pycuda.driver as cu
import pycuda.compiler as nvcc
from scipy.signal import convolve2d

def conv(K, P):
    S = np.sum(K * P)
    return S


def G_matrix(K, P):
    G = (K ** 2 + P ** 2) ** (1 / 2)
    return G
    
if __name__ == "__main__":

    try:
        name = 'lena_bw.png'
        img = Image.open(name, 'r')
        I = np.asarray(img)
        height = img.height
        width = img.width

        GX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        GY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        J = np.zeros([height, width])
        J0 = np.zeros([height, width])

        process_start = tm.process_time()
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                Gx = conv(I[i - 1:i + 2, j - 1:j + 2], GX)
                Gy = conv(I[i - 1:i + 2, j - 1:j + 2], GY)
                J0[i, j] = G_matrix(Gx, Gy)
        process_end = tm.process_time()
        print('Time spent in CPU (handmade convolution): {} seconds'.format(process_end - process_start))
        
        plt.imshow(J0)
        plt.show()
        
        process_start = tm.process_time()
        Gx = convolve2d(I.astype('float32'), GX, mode = 'same')
        Gy = convolve2d(I.astype('float32'), GY, mode = 'same')
        J = np.sqrt(Gx*Gx+Gy*Gy)

        J_s = (J - J.min()) / (J.max() - J.min())
        J_s = (J_s * 255).astype('uint8')

        process_end = tm.process_time()
        print('Time spent in CPU (existed function for convolution): {} seconds'.format(process_end - process_start))
        
        plt.imshow(J)
        plt.show()
        
        cu.init()
        d = cu.Device(1)
        ctx = d.make_context()

        kernel_size = 3
        block_size = (16, 16)
        #grid_size = calculate_grid_size((height, width), block_size)
        grid_size = (32, 32)
        #print(I.shape)
        #print(grid_size)

        I_gpu = cu.to_device(I.astype('float32'))
        J_gpu = cu.mem_alloc(J.nbytes)

        source = cu.module_from_file("sobel.cubin")

        kernel_naive = source.get_function("sobel_filter")
        kernel_naive.prepare(['P', 'P', 'Q', 'Q', 'Q', 'Q'])
        time = kernel_naive.prepared_timed_call(grid_size, block_size, I_gpu, J_gpu, height, width, kernel_size, 4)
        J1 = cu.from_device(J_gpu, shape=J.shape, dtype="float32")
        print("Time spent in kernel1: {}s".format(time() * 1e-3))

        print("L1 norm: {}".format( np.sum(np.sum( np.abs(J - J1) )) ))

        plt.imshow(J1)
        plt.show()
        
    finally:
        ctx.pop()
        print('\ndone')
