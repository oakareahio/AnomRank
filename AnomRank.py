import numpy as np

# structure only
def randomWalkWithRestart(M1, c=0.5, err_tolerance=0.0001):
    node_count = M1.shape[0]
    score = np.array([1/node_count]*node_count).reshape(node_count, 1)
    interim_result = []

    while True:
        prev_score = score.copy()
        score = (1-c)*M1.dot(score)

        err = np.abs(score-prev_score).sum()
        if err < err_tolerance:
            print(f'err : {err}')
            print(f'iteration count : {len(interim_result)}')
            break

        interim_result.append(score)

    r_cpi = np.sum(interim_result, axis=0)

    return r_cpi


def offsetScorePropagation(M1, M2, c=0.5, err_tolerance=0.0001):
    delta_M = M2 - M1
    r_old = randomWalkWithRestart(M1)
    score_offset = (1 - c) * delta_M.dot(r_old)
    r_offset = 0
    score = (1 - c) * M2.dot(score_offset)

    i = 0
    while True:
        prev_score = score.copy()
        score = (1 - c) * M2.dot(score)

        err = np.abs(score-prev_score).sum()
        if err < err_tolerance:
            print(f'err : {err}')
            print(f'iteration count : {i}')
            print(f'r_old : {r_old}')
            print(f'r_offset : {r_offset}')
            break

        r_offset += score
        i += 1

    r_osp = r_old + r_offset

    return r_osp


if __name__ == '__main__':
    M1 = np.array([[0, 1, 1. / 2, 0, 1. / 4, 1. / 2, 0],
                  [1. / 5, 0, 1. / 2, 1. / 3, 0, 0, 0],
                  [1. / 5, 0, 0, 1. / 3, 1. / 4, 0, 0],
                  [1. / 5, 0, 0, 0, 1. / 4, 0, 0],
                  [1. / 5, 0, 0, 1. / 3, 0, 1. / 2, 1],
                  [0, 0, 0, 0, 1. / 4, 0, 0],
                  [1. / 5, 0, 0, 0, 0, 0, 0]])
    M2 = np.array([[  0, 1,    0,    0, 1./3, 1./2, 0],
                  [1./5, 0,    0, 1./5,    0,    0, 0],
                  [1./5, 0,    0, 1./5, 1./3,    0, 0],
                  [1./5, 0,    0,    0, 1./3,    0, 0],
                  [1./5, 0, 1./2, 1./5,    0, 1./2, 0],
                  [   0, 0,    0, 1./5,    0,    0, 1],
                  [1./5, 0, 1./2, 1./5,    0,    0, 0]])
    M3 = np.array([[  0, 0, 1./3,    0, 1./4, 1./2, 0],
                  [   0, 0,    0,    0,    0,    0, 0],
                  [   0, 1,    0,    1, 1./4,    0, 0],
                  [   0, 0,    0,    0, 1./4,    0, 0],
                  [   0, 0, 1./3,    0,    0, 1./2, 0],
                  [1./2, 0,    0,    0,    0,    0, 0],
                  [1./2, 0, 1./3,    0, 1./4,    0, 0]])

    interim_result, r_cpi =randomWalkWithRestart(M1)
    print(r_cpi)
    r_osp =offsetScorePropagation(M1, M3)
    print(r_osp)