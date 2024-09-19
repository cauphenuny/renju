// author: Cauphenuny <https://cauphenuny.github.io/>
// date: 2024/07/26
#include <ctype.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include <stdint.h>

#include "util.h"
#include "board.h"
#include "game.h"

/**************************** manual ****************************/

int parse(char s[]) {
    if (isupper(s[0])) return s[0] - 'A' + 10;
    if (islower(s[0])) return s[0] - 'a' + 10;
    if (!isdigit(s[0])) return -1;
    int tmp = 0, i = 0;
    while (isdigit(s[i])) tmp = tmp * 10 + s[i] - '0', i++;
    return tmp;
}

point_t manual(const game_t game) {
    log("manual player%d", game.current_id);
    point_t pos;
    char input_x[2], input_y[2];
    log("waiting input. (format: 8 11 or 8 b or 8 B)");
    scanf("%s %s", input_x, input_y);
    pos.x = parse(input_x), pos.y = parse(input_y);
    log("%d, %d", pos.x, pos.y);
    return pos;
}
