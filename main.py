import skimage.color as color
import skimage.io as io
import PIL.Image as Image
import scipy.sparse as sparse
import scipy.linalg as la
import numpy as np
import pathlib
import time


def make_sparse_laplacian(lab, rad=1, activate_value=15):
    h, w = np.shape(lab)[:2]
    cnt_not_zeroes = ((2 * rad + 1) ** 2 - 1) * (w - 2 * rad) * (h - 2 * rad)

    data = np.zeros(cnt_not_zeroes)
    row = np.zeros(cnt_not_zeroes)
    col = np.zeros(cnt_not_zeroes)

    it = 0
    for i in range(0, h - rad):
        for j in range(0, w - rad):
            for k in range(rad, 2 * rad + 1):
                for p in range(rad, 2 * rad + 1):
                    if k == p:
                        continue
                    color_diff = ((lab[i - rad + k][j - rad + p][0] - lab[i][j][0]) ** 2 +
                                  (lab[i - rad + k][j - rad + p][1] - lab[i][j][1]) ** 2 +
                                  (lab[i - rad + k][j - rad + p][2] - lab[i][j][2]) ** 2) ** 0.5

                    if color_diff < activate_value:
                        continue

                    row[it] = i * w + j
                    col[it] = (i - rad + k) * w + (j - rad + p)
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
        elif fiedler_vector[i] == 0:
            c = [128, 128, 128]
        else:
            c = [255, 255, 255]
        dst[i // width][i % width] = c

    comb_image = np.zeros(dst.shape)
    comb_image[:, :, :, ] = (alpha * rgb[:, :, :, ] / 256) + ((1 - alpha) * dst[:, :, :, ] / 256)

    im = Image.fromarray((comb_image * 255).astype(np.uint8))
    im.save('Res/av-' + str(act_val) + '___now_time-' + str(time.time()).split('.')[0] +
            '___exec_time-' + str(dt)[:5] + "sec___v3___" + filename.split('\\')[1])


def solve_sparse(a, b):
    n = a.shape[0]
    x = np.random.rand(n)
    x /= la.norm(x)

    for i in range(10):
        y = a * x - b
        a_y = a * y
        t = - np.dot(y, a_y) / np.dot(a_y, a_y)
        z = x - t * y
        x = z

    return x


def calc_min_eig_vector(sparse_matrix):
    n = sparse_matrix.shape[0]
    x = np.random.randn(n)
    x /= la.norm(x)

    for i in range(10):
        y = solve_sparse(sparse_matrix, x)
        y /= la.norm(y)
        x = y

    return x


def calc_fiedler_vector(filename, act_val):
    rgb = io.imread(filename)
    lab = color.rgb2lab(rgb)
    sparse_laplacian = make_sparse_laplacian(lab, activate_value=act_val)

    return calc_min_eig_vector(sparse_laplacian)


if __name__ == '__main__':
    dir_name = 'C:\\Users\\Arseny\\Documents\\Prog\\PythonProjects\\MinCut\\Ref'
    path_list = pathlib.Path(dir_name).glob('**/trees_and_sun.jpg')
    for path in path_list:
        file_name = "Ref\\" + str(path).split('\\Ref\\')[1]

        color_activate_value = 10

        start_time = time.time()
        fiedler_vec = calc_fiedler_vector(file_name, color_activate_value)
        end_time = time.time()

        save_concat_image(file_name, fiedler_vec, end_time - start_time, color_activate_value)
