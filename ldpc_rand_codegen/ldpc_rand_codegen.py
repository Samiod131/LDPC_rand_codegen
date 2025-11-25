from random import sample, shuffle

import numpy as np
import scipy


'''
This code generates random LDPC codes using the curvebal method for row and column exchange, 
and the reverse-Cuthill-McKee algorithm to reduce the bandwidth of the adjacency matrix.

These codes, while random, are slightly optimized for the decoder by reducing the bandwidth 
of the adjacency matrix.

The generation of these codes is explained in details in chapter 
'''


'''
Curveball method directly taken from
https://www.nature.com/articles/ncomms5114
'''


def find_presences(input_matrix):
    # SD : Returns positions of all 1 values in columns or rows
    num_rows, num_cols = input_matrix.shape
    hp = []
    iters = num_rows if num_cols >= num_rows else num_cols
    input_matrix_b = input_matrix if num_cols >= num_rows else np.transpose(
        input_matrix)
    for r in range(iters):
        hp.append(list(np.where(input_matrix_b[r] == 1)[0]))
    return hp


def curve_ball(input_matrix, r_hp, num_iterations=-1):
    num_rows, num_cols = input_matrix.shape
    l = range(len(r_hp))
    num_iters = 10*min(num_rows, num_cols) if num_iterations == - \
        1 else num_iterations
    for rep in range(num_iters):
        AB = sample(l, 2)
        a = AB[0]
        b = AB[1]
        ab = set(r_hp[a]) & set(r_hp[b])  # common elements
        l_ab = len(ab)
        l_a = len(r_hp[a])
        l_b = len(r_hp[b])
        if l_ab not in [l_a, l_b]:
            tot = list(set(r_hp[a]+r_hp[b])-ab)
            ab = list(ab)
            shuffle(tot)
            L = l_a-l_ab
            r_hp[a] = ab+tot[:L]
            r_hp[b] = ab+tot[L:]
    out_mat = np.zeros(input_matrix.shape, dtype='int8') if num_cols >= num_rows else np.zeros(
        input_matrix.T.shape, dtype='int8')
    for r in range(min(num_rows, num_cols)):
        out_mat[r, r_hp[r]] = 1
    result = out_mat if num_cols >= num_rows else out_mat.T
    return result


'''
Bipartite graph reverse-Cuthill-McKee

Reduces the bandwidth of any bipartite graph in the form of its partial 
adjacency matrix.
'''


def gen_permute_mat(vector):
    '''
    Generates a permutation matrix from a permutation array given by the scipy
    rcmk function.
    '''
    size = len(vector)
    per_mat = np.zeros((size, size))

    for row, _ in enumerate(vector):
        per_mat[row, vector[row]] = 1

    return per_mat


def is_redundant(matrix):
    '''
    Checks if two checks are redundant in a parity-check matrix.
    '''
    fail = False
    for i in range(matrix.shape[0]-1):
        if np.allclose(matrix[i, :], matrix[i+1, :]):
            fail = True

    return fail


def checks_rcmk_run(matrix):
    '''
    Permutes the checks of the parity check matrix to reduce bandwith.
    '''
    # Matrix for check exchange
    checks_mat = np.matmul(matrix, np.transpose(matrix))
    check_graph = scipy.sparse.csr_matrix(checks_mat)

    perm_vect_check = scipy.sparse.csgraph.reverse_cuthill_mckee(check_graph, symmetric_mode=True)
    check_permute_mat = gen_permute_mat(perm_vect_check)

    new_mat = np.matmul(check_permute_mat, matrix)

    return new_mat


def bits_rcmk_run(matrix):
    '''
    Permutes the bits of the parity check matrix to reduce bandwith.
    '''
    # matrix for bit exchange
    bit_mat = np.matmul(np.transpose(matrix), matrix)
    bit_graph = scipy.sparse.csr_matrix(bit_mat)

    perm_vect_bit = scipy.sparse.csgraph.reverse_cuthill_mckee(bit_graph, symmetric_mode=True)
    bit_permute_mat = np.transpose(gen_permute_mat(perm_vect_bit))

    new_mat = np.matmul(matrix, bit_permute_mat)

    return new_mat


def is_antitrace(matrix, block_size=3):
    """
    Evaluates if the given matrix is closer to a trace or antitrace matrix.
    """
    anti_trace = None
    up_left = np.sum(matrix[0:block_size, 0:block_size])
    up_right = np.sum(matrix[0:block_size, -block_size:])
    down_left = np.sum(matrix[-block_size:, 0:block_size])
    down_right = np.sum(matrix[-block_size:, -block_size:])

    if up_left+down_right < up_right+down_left:
        anti_trace = True
    elif up_left+down_right > up_right+down_left:
        anti_trace = False

    return anti_trace


def reverse_checks(matrix):
    '''
    If the matrix given as an anti-reduced bandwidth, inverse the columns order.
    '''
    new_mat = np.zeros((matrix.shape))

    mat_antitrace = None
    bs = 1
    max_bs = min(matrix.shape)/2

    while mat_antitrace is None:
        if bs > max_bs:
            mat_antitrace = False
        else:
            mat_antitrace = is_antitrace(matrix, block_size=bs)
            bs += 1

    if mat_antitrace:
        for i in range(len(matrix[0, :])):
            new_mat[:, -i-1] = matrix[:, i]
    else:
        new_mat = matrix

    return new_mat


def run_bipartite_rcmk(matrix):
    '''
    Reduces the bandwidth of a parity check matrix given.
    '''
    newmat = bits_rcmk_run(matrix)
    newmat = checks_rcmk_run(newmat)
    newmat = reverse_checks(newmat)

    return newmat


def get_rand_code(bit_deg=3, check_deg=4, nmult=2, rcmk=True):
    '''
    Generates a random ldpc parity check matrix with the right dimensions, then 
    reduces the bandwidth of it.
    '''
    # Generates a full connected block matrix for a basic unit
    block_mat = np.ones((bit_deg, check_deg))

    if nmult == 1:
        parity = block_mat
    else:
        # Creates a block diagonal matrix depending on total code size
        # This way each line of a matrix is a check
        square_mat_id = np.kron(np.eye(nmult, dtype=int), block_mat)

        # Mix the rows and columns with the curveball method
        r_hp = find_presences(square_mat_id)
        parity = curve_ball(square_mat_id, r_hp)

        while is_redundant(parity):
            parity = curve_ball(square_mat_id, r_hp)

        if rcmk:
            parity= run_bipartite_rcmk(parity)

    return parity


if __name__ == "__main__":
    parity_check = get_rand_code(
        bit_deg=3, check_deg=4, nmult=2, rcmk=True)