// author: Cauphenuny
// date: 2024/07/26
#include <ctype.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include <stdint.h>

#include "util.h"
#include "game.h"
#include "board.h"

int parse(char s[], bool is_char)
{
    if (is_char) {
        if (isupper(s[0])) return s[0] - 'A';
        if (islower(s[0])) return s[0] - 'a';
        return -1;
    } else {
        if (!isdigit(s[0])) return -1;
        int tmp = 0, i = 0;
        while (isdigit(s[i])) tmp = tmp * 10 + s[i] - '0', i++;
        tmp--;
        return tmp;
    }
}

point_t manual(const game_t game, void* assets) {
    point_t pos;
    char input_x[10], input_y[10];
    log_i("waiting input. (format: H 8 or h 8)");
    prompt();
    scanf("%s %s", input_x, input_y);
    pos.y = parse(input_x, 1), pos.x = parse(input_y, 0);
    return pos;
}
