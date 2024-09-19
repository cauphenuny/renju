#pragma once

#include <stdbool.h>
#include <stdint.h>

#define BOARD_SIZE 15

#define WIN_LENGTH 5

typedef struct {
    int8_t x, y;
} point_t;

typedef int board_t[BOARD_SIZE][BOARD_SIZE];

enum {
    POS_ACCEPT,
    POS_BANNED_LONG,
    POS_BANNED_33,
    POS_BANNED_44,
};

extern const char* POS_BAN_MSG[];

void print(const board_t);
void wrap_area(const board_t, int*, int*, int*, int*, int);
bool inboard(point_t);
char id2name(int);
void refresh(board_t board);
bool available(board_t board, point_t pos);
void put(board_t board, int id, point_t pos);
int  check_draw(const board_t);
int  check(const board_t, point_t);
int banned(const board_t, point_t, int);
void test_ban(void);