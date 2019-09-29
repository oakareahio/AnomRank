import numpy as np


def randomWalkWithRestart(transition_matrix, init_score, c=0.1, err_tolerance=0.0001):
    score = init_score
    interim_result = []
    interim_result.append(c * init_score)
    while True:
        prev_score = score.copy()
        score = (1 - c) * transition_matrix.dot(score)

        err = np.abs(score-prev_score).sum()
        if err < err_tolerance:
            print(f'err : {err}')
            print(f'iteration count : {len(interim_result)}')
            break
        interim_result.append(score)

    r_rwr = np.sum(interim_result, axis=0)
    return r_rwr


def offsetScorePropagtion(transition_M_before, transition_M_after, r_old, score_type,
                          delta_init_score=None, c=0.1, err_tolerance=0.0001):
    delta_M = transition_M_after - transition_M_before
    score_offset = (1 - c) * delta_M.dot(r_old)
    score = score_offset
    interim_result = []
    interim_result.append(score_offset)

    i = 0
    while True:
        prev_score = score.copy()
        if score_type == 'structure':
            score = (1 - c) * transition_M_after.dot(score)
        elif score_type == 'weight':
            score = (1 - c) * transition_M_after.dot(score) + c * transition_M_after.dot(delta_init_score)
        err = np.abs(score - prev_score).sum()
        if err < err_tolerance:
            r_offset = np.sum(interim_result, axis=0)
            print(f'err : {err}')
            print(f'iteration count : {i}')
            print(f'r_old : {r_old}')
            print(f'r_offset : {r_offset}')
            break
        interim_result.append(score)
        i += 1
    r_osp = r_old + r_offset
    return r_osp


def randomWalkWithRestartS(M1):
    node_count = M1.shape[0]
    init_score = np.array([1/node_count]*node_count).reshape(node_count, 1)
    r_s = randomWalkWithRestart(M1, init_score)
    return r_s


def offsetScorePropagationS(M_before, M_after):
    r_old = randomWalkWithRestartS(M_before)
    r_osp = offsetScorePropagtion(M_before, M_after, r_old, 'structure')
    return r_osp


def differentiateScore(p_new, p_now, p_old):
    p_diff1 = p_new - p_now
    p_diff2 = (p_new - p_now) - (p_now - p_old)
    return [p_diff1, p_diff2]


def AnomScoreS(m_lst: list):
    p_old = np.zeros((m_lst[0].shape[0], 1))
    p_now = np.zeros((m_lst[0].shape[0], 1))

    score_lst=[]
    for i in range(len(m_lst)):
        if i+1 < len(m_lst):
            m_before = m_lst[i]
            m_after = m_lst[i+1]
        else:
            break
        p_new = offsetScorePropagationS(m_before, m_after)
        score = differentiateScore(p_new, p_now, p_old)
        score_lst.append(score)
        p_old = p_now
        p_now = p_new

    last_score = np.array(score_lst[-1])
    abs_last_score = np.abs(last_score)
    anomScore = np.max(abs_last_score, axis=0)

    return anomScore


def randomWalkWithRestartW(norm_M1, init_score):
    r_w = randomWalkWithRestart(norm_M1, init_score)
    return r_w


def getInitialScore(weighed_M):
    total_edge_weight = weighed_M.sum()
    total_node_edge_weight = weighed_M.sum(axis=1)
    init_score = (total_node_edge_weight/total_edge_weight).reshape(weighed_M.shape[0], 1)
    return init_score


def offsetScorePropagationW(norm_M1, norm_M2, weighed_M1, weighed_M2):
    init_score1 = getInitialScore(weighed_M1)
    init_score2 = getInitialScore(weighed_M2)
    delta_init_score = init_score2 - init_score1

    r_old = randomWalkWithRestartW(norm_M1, init_score1)
    r_osp = offsetScorePropagtion(norm_M1, norm_M2, r_old, 'weight', delta_init_score)
    return r_osp


def AnomScoreW(m_norm_lst, m_weighed_lst):
    p_old = np.zeros((m_norm_lst[0].shape[0], 1))
    p_now = np.zeros((m_norm_lst[0].shape[0], 1))

    score_lst=[]
    for i in range(len(m_norm_lst)):
        if i+1 < len(m_norm_lst):
            m_norm_before = m_norm_lst[i]
            m_norm_after = m_norm_lst[i+1]
        else:
            break
        p_new = offsetScorePropagationW(m_norm_before, m_norm_after, m_weighed_lst[i], m_weighed_lst[i+1])
        score = differentiateScore(p_new, p_now, p_old)
        score_lst.append(score)
        p_old = p_now
        p_now = p_new

    last_score = np.array(score_lst[-1])
    abs_last_score = np.abs(last_score)
    anomscoreW = np.max(abs_last_score, axis=0)
    return anomscoreW



if __name__ == '__main__':

    M1 = np.array([[  0, 1./5, 1./5, 1./5, 1./5,    0, 1./5],
                  [   1,    0,    0,    0,    0,    0,    0],
                  [1./2, 1./2,    0,    0,    0,    0,    0],
                  [   0, 1./3, 1./3,    0, 1./3,    0,    0],
                  [1./4,    0, 1./4, 1./4,    0, 1./4,    0],
                  [1./2,    0,    0,    0, 1./2,    0,    0],
                  [   0,    0,    0,    0,    1,    0,    0]])

    M2 = np.array([[  0, 1./3,    0, 1./3,    0,    0, 1./3],
                  [   1,    0,    0,    0,    0,    0,    0],
                  [1./5, 1./5,    0,    0, 1./5, 1./5, 1./5],
                  [   0, 1./2,    0,    0,    0,    0, 1./2],
                  [1./3,    0, 1./3, 1./3,    0,    0,    0],
                  [1./3,    0,    0,    0, 1./3,    0, 1./3],
                  [   0,    0,    0,    0,    0,    1,    0]])

    M3 = np.array([[  0,    0,    0,    0,    0, 1./2, 1./2],
                  [   0,    0,    1,    0,    0,    0,    0],
                  [1./3,    0,    0,    0, 1./3,    0, 1./3],
                  [   0,    0,    1,    0,    0,    0,    0],
                  [1./4,    0, 1./4, 1./4,    0,    0, 1./4],
                  [1./2,    0,    0,    0, 1./2,    0,    0],
                  [   0,    1,    0,    0,    0,    0,    0]])

    M_lst = [M1.T, M2.T, M3.T, M2.T, M1.T, M3.T, M2.T]
    anomScoreS = AnomScoreS(M_lst)
    print(anomScoreS)

    M1 = np.array([[0, 5, 3, 1, 9, 0, 2],
                  [ 8, 0, 0, 0, 0, 0, 0],
                  [ 4, 3, 0, 0, 0, 0, 0],
                  [ 0, 1, 1, 0, 1, 0, 0],
                  [ 2, 0, 6, 4, 0, 1, 1],
                  [ 3, 0, 0, 0, 3, 0, 0],
                  [ 0, 0, 0, 0, 1, 0, 0]])
    M2 = np.array([[0, 4, 0, 1, 0, 0, 2],
                  [ 3, 0, 0, 0, 0, 0, 0],
                  [ 3, 3, 0, 0, 3, 3, 3],
                  [ 0, 1, 0, 0, 0, 0, 8],
                  [ 2, 0, 2, 2, 0, 0, 0],
                  [ 3, 0, 0, 0, 3, 0, 9],
                  [ 0, 0, 0, 0, 1, 0, 0]])
    M3 = np.array([[0, 0, 0, 0, 0, 5, 7],
                  [ 0, 0, 2, 0, 0, 0, 0],
                  [ 3, 0, 0, 0, 2, 0, 1],
                  [ 0, 0, 4, 0, 0, 0, 0],
                  [ 2, 0, 2, 2, 0, 0, 2],
                  [ 3, 0, 0, 0, 6, 0, 0],
                  [ 0, 500, 0, 0, 0, 0, 0]])

    M1_norm = M1/M1.sum(axis=1)[:, None]
    M2_norm = M2/M2.sum(axis=1)[:, None]
    M3_norm = M3/M3.sum(axis=1)[:, None]

    M_norm_lst = [M1_norm.T, M2_norm.T, M3_norm.T,
                  M2_norm.T, M1_norm.T, M3_norm.T,
                  M2_norm.T]
    M_lst = [M1, M2, M3,
             M2, M1, M3,
             M2]
    anomScoreW = AnomScoreW(M_norm_lst, M_lst)
    print(anomScoreS)
    print(anomScoreW)