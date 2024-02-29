import pathlib
import time

import PIL.Image as Image
import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
import skimage.color as color
import skimage.io as io

ITER1 = 50
ITER2 = 20


def make_sparse_laplacian(grayscale, activate_value=15):
    h, w = np.shape(grayscale)[:2]
    cnt_not_zeroes = 8 * (w - 2) * (h - 2)

    data = np.zeros(cnt_not_zeroes)
    row = np.zeros(cnt_not_zeroes)
    col = np.zeros(cnt_not_zeroes)

    it = 0
    for i in range(0, h - 1):
        for j in range(0, w - 1):
            for k in range(0, 2):
                for p in range(0, 2):
                    if k == 0 and p == 0:
                        continue

                    if abs(grayscale[i + k][j + p] - grayscale[i][j]) < activate_value:
                        continue

                    row[it] = i * w + j
                    col[it] = (i + k) * w + (j + p)
                    data[it] = 1
                    it += 1

                    row[it] = col[it - 1]
                    col[it] = row[it - 1]
                    data[it] = 1
                    it += 1

    adjacency_matrix = sparse.csr_matrix((data, (row, col)), shape=(w * h, w * h))
    e_vec = np.full((h * w), fill_value=1)
    degree_vec = adjacency_matrix * e_vec

    degree_row = np.empty(w * h)
    for i in range(w * h):
        degree_row[i] = i

    degree_matrix = sparse.csr_matrix((degree_vec, (degree_row, degree_row)), shape=(w * h, w * h))
    return degree_matrix - adjacency_matrix


def save_concat_image(filename, fiedler_vector, dt, act_val, alpha=0.15):
    rgb = io.imread(filename)
    height, width = rgb.shape[:2]

    dst = np.zeros(rgb.shape)
    for i in range(fiedler_vector.size):
        if fiedler_vector[i] > 0:
            c = [0, 0, 0]
        else:
            c = [255, 255, 255]
        dst[i // width][i % width] = c

    comb_image = np.zeros(dst.shape)
    comb_image[:, :, :, ] = (alpha * rgb[:, :, :, ] / 256) + ((1 - alpha) * dst[:, :, :, ] / 256)

    im = Image.fromarray((comb_image * 255).astype(np.uint8))
    im.save('Res/moscow/av-' + str(act_val) + '___now_time-' + str(time.time()).split('.')[0] +
            '___exec_time-' + str(dt)[:5] + "sec___v9_" + str(ITER1) + "_" + str(ITER2) + "___" + filename.split('\\')[1])


def solve_sparse(a, b):
    n = a.shape[0]
    x = np.random.rand(n)
    x /= la.norm(x)

    for i in range(ITER1):
        y = b - a * x
        a_y = a * y
        t = np.dot(y, a_y) / np.dot(a_y, a_y)
        z = x - t * y
        x = z

    return x


def calc_min_eig_vector(sparse_matrix, filename):
    n = sparse_matrix.shape[0]
    x = np.random.randn(n)
    x /= la.norm(x)

    for i in range(ITER2):
        y = solve_sparse(sparse_matrix, x)
        y /= la.norm(y)
        x = y
        save_concat_image(filename, x, i, 0)

    return x


def calc_fiedler_vector(filename, act_val):
    rgb = io.imread(filename)
    lab = color.rgb2gray(rgb)
    sparse_laplacian = make_sparse_laplacian(lab, activate_value=act_val)

    return calc_min_eig_vector(sparse_laplacian, filename)


def open_cv_edge_detection(filename):
    import cv2 as cv
    from matplotlib import pyplot as plt
    img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    edges = cv.Canny(img, 100, 200)
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':
    color_activate_value = 0.007

    dir_name = 'C:\\Users\\Arseny\\Documents\\Prog\\PythonProjects\\MinCut\\Ref'
    path_list = pathlib.Path(dir_name).glob('**/moscow-city.jpg')
    for path in path_list:
        file_name = "Ref\\" + str(path).split('\\Ref\\')[1]

        # start_time = time.time()
        # open_cv_edge_detection(file_name)
        # end_time = time.time()
        #
        # print(end_time - start_time)

        start_time = time.time()
        fiedler_vec = calc_fiedler_vector(file_name, color_activate_value)
        end_time = time.time()

        save_concat_image(file_name, fiedler_vec, end_time - start_time, color_activate_value)

# TODO: try edge detection (not object) for this mode (HSV)
# TODO: delete noise
# TODO: save image without noise
# TODO: try another modes and metrics for objects and edges
# TODO: recursive chain
