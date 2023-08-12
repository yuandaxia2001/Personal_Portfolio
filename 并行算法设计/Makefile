CXXFLAGS+=-O2 -ggdb -DDEBUG
CXXFLAGS+=-Wall -Wextra

all: sudoku_solve

sudoku_solve: main.c sudoku.cc -lpthread
	g++ -O2 -o $@ $^ 
