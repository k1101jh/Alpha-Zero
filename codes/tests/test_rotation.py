import numpy as np


def test_rotation():
    board_size = 3
    states = np.array([[[[1, 2, 0],
                         [2, 1, 0],
                         [0, 1, 2]]],
                       [[[0, 3, 4],
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
            rotated_state = np.rot90(state, i, axes=(1, 2))
            rotated_visit_count = np.rot90(visit_count.reshape(board_size, board_size), i, axes=(0, 1))
            new_states.append(rotated_state)
            new_visit_counts.append(rotated_visit_count)

            new_states.append(np.flip(rotated_state, 2))
            new_visit_counts.append(np.fliplr(rotated_visit_count))

    for i, (state, visit_count) in enumerate(zip(new_states, new_visit_counts)):
        print("case: ", i)
        for i in range(3):
            for j in range(3):
                print(state[0][i][j], end=' ')
            print('')
        print('')
        for i in range(3):
            for j in range(3):
                print(visit_count[i][j], end=' ')
            print('')
        print('')


if __name__ == "__main__":
    test_rotation()
