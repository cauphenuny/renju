// author: Cauphenuny <https://cauphenuny.github.io/>
// date: 2024/07/26
#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <stdint.h>

#include "util.h"
#include "board.h"

/**************************** manual ****************************/

int parse(char s[]) {
    if (isdigit(s[0])) return s[0] - '0';
    if (isupper(s[0])) return s[0] - 'A' + 10;
    if (islower(s[0])) return s[0] - 'a' + 10;
    return -1;
}

point_t manual(const board_t board) {
    point_t pos;
    char input_x[2], input_y[2];
    do {
        log("waiting input.");
        scanf("%s %s", input_x, input_y);
        pos.x = parse(input_x), pos.y = parse(input_y);
    } while ((!inboard(pos) || board[pos.x][pos.y]) && loge("invalid input!"));
    return pos;
}
