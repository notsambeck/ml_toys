'''
minesweeper for console use
'''
import numpy as np
from timeit import default_timer


class Minesweeper:
    def __init__(self, rows=10, cols=8, mines=None):
        self.rows = rows
        self.cols = cols

        # board is rows * cols
        self._state = np.zeros((4, rows, cols), dtype='uint8')
        self.mines = self._state[0]
        self.counts = self._state[1]
        self.viewed = self._state[2]
        self.flags = self._state[3]

        if mines is None:
            self.mine_qty = rows * cols // 8
        else:
            assert isinstance(mines, int)
            self.mine_qty = mines

        mines_on_board = 0

        # place mines on board at random
        while mines_on_board < self.mine_qty:
            y = np.random.randint(0, rows)
            x = np.random.randint(0, cols)
            if self.mines[y, x] == 0:
                self.mines[y, x] = 1
                mines_on_board += 1

        # build counts (numbers that show on each square when viewed)
        for row in range(rows):
            for col in range(cols):
                if not self.mines[row, col]:
                    self.counts[row, col] = self.mine_count(row, col)
                # otherwise zero

        self.start_time = default_timer()
        self.play()

    def show_board(self):
        print('time: {:.0f} seconds'.format(default_timer() - self.start_time))
        print('mines: {}  flags: {}'.format(self.mine_qty, np.sum(self.flags)))
        print()
        print('* ' + ' '.join([str(i) for i in range(self.cols)]) + ' *')
        for row in range(self.rows):
            current_row = []
            for col in range(self.cols):
                # if flagged, show flag
                if self.flags[row, col]:
                    current_row.append('x')

                # if checked, show count
                elif self.viewed[row, col]:
                    count = self.counts[row, col]
                    if count:
                        current_row.append(str(count))
                    else:
                        current_row.append('.')

                # otherwise blank
                else:
                    current_row.append(' ')

            # print row#, assembled row, border
            print(' '.join([str(row)] + current_row + ['*']))

        # border
        print('* ' * (self.cols + 2))

    def go(self, row, col):
        # make a move; return True if game continues
        if not self.valid_loc(row, col):
            print('invalid location')
            return True

        if self.viewed[row, col] == 1:
            print('repeat move is invalid')
            return True

        # for valid moves:
        self.viewed[row, col] = 1

        # if mine
        if self.mines[row, col]:
            print('you lose!')
            return False

        # if counts == 0: safe square, expand it
        if self.counts[row, col] == 0:
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    new_col = col + i
                    new_row = row + j
                    if self.valid_loc(new_row, new_col) and not self.viewed[new_row, new_col]:
                        self.go(new_row, new_col)

        return True

    def valid_loc(self, row, col):
        return (0 <= col < self.cols) and (0 <= row < self.rows)

    def mine_count(self, row, col):
        # count mines in adjacent squares
        count = 0
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == j == 0:
                    continue
                # print(count)
                # print(x + i, y + j)
                # print(self.board[x + i, y + j])
                new_row, new_col = row + i, col + j
                if self.valid_loc(new_row, new_col):
                    count += self.mines[new_row, new_col]
        return count

    def play(self):
        game_on = True
        win = False
        while game_on:
            self.show_board()
            f = input('for a flag, enter f or . ')
            r = input('row: ')
            c = input('col: ')
            try:
                r = int(r)
                c = int(c)
            except ValueError:
                print('numbers fool')
                continue

            if f == 'f' or f == '.' or f == 'F':
                self.flags[r, c] = 1 - self.flags[r, c]
            else:
                game_on = self.go(r, c)

            if np.array_equal(self.flags, self.mines):
                win = True
                break
            if np.array_equal(self.viewed, np.ones((self.rows, self.cols), 'uint8') - self.mines):
                win = True
                break
            else:
                continue

        if win:
            print('DUDE!')
            print()
            print('   #######    ')
            print('  ##     ##   ')
            print('  |[o] [o]|   ')
            print('  |   A   |   ')
            print('  \  ===  /   ')
            print('    ----      ')
        else:
            print('*wanh wah wah*')
            print('   ______')
            print('  /      \ ')
            print('  | x   x |')
            print('  |   o   |')
            print('   \     /')
            print('    ====')
            print('    \__/')
            print('game over; {} mines left'.format(np.sum((self.mines - self.flags).clip(0))))

        replay = input('enter to replay')

        if 'n' in replay.lower():
            return None
        self.__init__()


if __name__ == '__main__':
    m = Minesweeper()
