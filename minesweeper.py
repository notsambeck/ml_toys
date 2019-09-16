'''
minesweeper in the console... works pretty ok
'''
import numpy as np
from timeit import default_timer


class Minesweeper:
    """class represents the game & associated metadata"""
    def __init__(self, rows=10, cols=8, mines=None):
        self.rows = rows
        self.cols = cols

        # board is rows * cols; game state is stored in self._state[0:4] as ndarray
        # also aliased as self.mines, etc. for convenience
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
                    self.counts[row, col] = self.count_mines(row, col)
            # otherwise zero
        self.start_time = default_timer()
        self.play()

    def show_board(self):
        """print board to console, does not affect game state"""
        print('timer running: {:.0f} seconds'.format(default_timer() - self.start_time))
        print('board: mines={}  used_flags={}'.format(self.mine_qty, np.sum(self.flags)))
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

    def probe(self, row, col):
        """test point (row, col); return True if game continues, else False"""
        if not self.valid_loc(row, col):
            print('invalid location')
            return True

        if self.viewed[row, col] == 1:
            print('repeat move is invalid')
            return True

        # for valid moves, mark point as viewed:
        self.viewed[row, col] = 1

        # if rol, col is a mine
        if self.mines[row, col]:
            print('you lose!')
            return False

        # if counts == 0: safe square, expand it recursively
        if self.counts[row, col] == 0:
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    new_col = col + i
                    new_row = row + j
                    if self.valid_loc(new_row, new_col) and not self.viewed[new_row, new_col]:
                        self.probe(new_row, new_col)

        return True

    def valid_loc(self, row, col):
        return (0 <= col < self.cols) and (0 <= row < self.rows)

    def count_mines(self, row, col):
        """count mines in squares adjacent to (row, col)"""
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
        """get player input and change game state"""
        game_on = True   # whether game continues
        win = False
        while game_on:
            self.show_board()
            print('Enter coordinate for next move:')
            c = input('x_coordinate 0-{}: '.format(self.cols - 1))
            r = input('y_coordinate 0-{}: '.format(self.rows - 1))
            try:
                r = int(r)
                c = int(c)
            except ValueError:
                print('\n' * 8)
                input('no, use numbers fool')
                continue

            # get  move type
            print('''
Select move, then press enter. 

    probe       <blank>
    flag        move_type
    back        b
''')
            move_type = input()
            
            if move_type.lower() in ['f,' 'flag', '.']:
                self.flags[r, c] = 1 - self.flags[r, c]
            elif move_type.lower() in ['b', 'back']:
                continue
            else:
                game_on = self.probe(r, c)

            if np.array_equal(self.flags, self.mines):
                win = True
                break

            if np.array_equal(self.viewed, np.ones((self.rows, self.cols), 'uint8') - self.mines):
                win = True
                break
            else:
                continue

        if win:
            print('you won eh')
            print()
            print('   #######    ')
            print('  ##     ##   ')
            print('  |[o] [o]|   ')
            print('  |   A   |   ')
            print('  \  ===  /   ')
            print('    ----      ')
        else:
            print('*wanh wah wah you are dead*')
            print('   ______  ')
            print('  /      \ ')
            print('  | x   x |')
            print('  |   n   |')
            print('   \     / ')
            print('    ====   ')
            print('    \__/   ')
            print()
            print('game over eh')
        
        print()
        replay = input('press enter to replay...')

        if 'n' in replay.lower():
            return None
        self.__init__()


if __name__ == '__main__':
    m = Minesweeper()
