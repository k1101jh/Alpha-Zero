import numpy as np


def test_rotation():
    board_size = 3
    states = np.array([[[[1, 2, 0],
                         [2, 1, 0],
                         [0, 1, 2]]],
                       [[[0, 1, 2],
                         [0, 0, 0],
                         [2, 1, 0]]]])
    visit_counts = np.array([[0, 0, 3,
                              0, 0, 2,
                              1, 0, 0],
                             [5, 0, 0,
                              6, 7, 8,
                              0, 0, 9]])

    new_states = []
    new_visit_counts = []

    for state, visit_count in zip(states, visit_counts):
        for i in range(4):
            new_states.append(np.rot90(state, i, axes=(1, 2)))
            new_visit_counts.append(np.rot90(visit_count.reshape(board_size, board_size), i).flatten())

        new_states.append(np.flip(state, 2))
        new_visit_counts.append(np.fliplr(visit_count.reshape(board_size, board_size)).flatten())

    for i, (state, visit_count) in enumerate(zip(new_states, new_visit_counts)):
        print("case: ", i)
        for i in range(3):
            for j in range(3):
                print(state[0][i][j], end=' ')
            print('')
        print('')
        for i in range(3):
            for j in range(3):
                print(visit_count[i * 3 + j], end=' ')
            print('')
        print('')


if __name__ == "__main__":
    test_rotation()
