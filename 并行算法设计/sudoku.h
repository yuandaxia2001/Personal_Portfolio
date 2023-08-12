#ifndef SUDOKU_H
#define SUDOKU_H

const int NUM = 9;
enum { ROW=9, COL=9, N = 81, NEIGHBOR = 20 };
bool solve_sudoku_dancing_links(int board[81]);
void input(const char in[N],int board[81]);
#endif
