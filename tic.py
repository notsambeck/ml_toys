'''
tic tac toe
'''
from random import randint
from copy import deepcopy


class Tic:
    def __init__(self):
        self.players = ['X', 'O']
        self.whos_turn = 0
        self.board = [[0 for i in range(3)] for j in range(3)]
        self.winner = None
        self.stalemate = False

    def play(self):
        for player in self.players:
            self.show_board()
            print('Player: {}'.format(player))
            while True:
                r = input('row 1-3: ')
                c = input('col 1-3: ')
                try:
                    row = int(r) - 1
                    col = int(c) - 1
                    # TODO: test for 1-3
                    assert self.board[row][col] == 0
                    break

                except ValueError:
                    print('enter numbers 1-3 for row and column')
                except AssertionError:
                    print('you cannot go there again')

            self.board[row][col] = player
            self.check_victory()

            if self.winner:
                print('{} wins!'.format(self.winner))
                return self.winner
            elif self.stalemate:
                print('stalemate')
                return None

        self.play()

    def play_computer(self):
        self.show_board()
        player = self.players[self.whos_turn]
        print('Player: {}'.format(player))
        if not self.whos_turn:
            while True:
                r = input('row 1-3: ')
                c = input('col 1-3: ')
                try:
                    r = int(r) - 1
                    c = int(c) - 1
                    # TODO: test for 1-3
                    assert self.board[r][c] == 0
                    break

                except ValueError:
                    print('enter numbers 1-3 for row and column')
                except AssertionError:
                    print('you cannot go there again')

        else:
            tries = 0
            while True:
                r = randint(0, 2)
                c = randint(0, 2)
                try:
                    tries += 1
                    print(r, c)
                    for row in self.board:
                        print(row)

                    # always make legal move
                    assert self.board[r][c] == 0

                    # try to lose up to 20x
                    if tries < 20:
                        try_this_board = deepcopy(self.board)
                        try_this_board[r][c] = player
                        assert self.check_victory(try_this_board) is None
                    break

                except AssertionError:
                    print('not a good move')
                    continue

        # print(r, '?', c)
        self.board[r][c] = player
        self.check_victory()

        if self.winner:
            print('{} wins!'.format(self.winner))
            return self.winner

        elif self.stalemate:
            print('stalemate')
            return None

        self.whos_turn = not self.whos_turn
        self.play_computer()

    def check_victory(self, board=None):
        # there are 3 row wins, 3 column wins, and 2 diagonals;
        # check each of them
        # return winner if there is a winner, else return none
        if board is None:
            board = self.board

        for row in board:
            if row[0] and row[0] == row[1] == row[2]:
                self.winner = row[0]
                return self.winner

        for col in board[:][0], board[:][1], board[:][2]:
            if col[0] and col[0] == col[1] == col[2]:
                self.winner = col[0]
                return self.winner

        if board[0][0]:
            if board[0][0] == board[1][1] == board[2][2]:
                self.winner = board[0][0]
                return self.winner

        if board[0][2]:
            if board[0][2] == board[1][1] == board[2][0]:
                self.winner = board[0][2]
                return self.winner

        # check stalemate
        for row in board:
            for el in row:
                if not el:
                    return None

        # if we haven't returned yet, it's a stalemate.
        self.stalemate = True
        return 'stalemate'

    def test_check_victory(self):
        test_board = [[2, 2, 1],
                      [0, 1, 0],
                      [1, 0, 2]]
        assert self.check_victory(test_board) == 1

        test_board = [[2, 2, 1],
                      [1, 0, 2],
                      [2, 1, 2]]
        assert self.check_victory(test_board) is None

        test_board = [[2, 2, 1],
                      [1, 1, 2],
                      [2, 1, 2]]
        assert self.check_victory(test_board) == 'stalemate'

        print('victory tests passed')
        # clear
        self.__init__()

    def show_board(self):
        print()
        for row in self.board:
            print(row)
