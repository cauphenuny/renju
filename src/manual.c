// author: Cauphenuny
// date: 2024/07/26
#include "board.h"
#include "game.h"
#include "util.h"

#include <assert.h>
#include <ctype.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int parse(char s[])
{
    if (strcmp(s, "re") == 0 || strcmp(s, "regret") == 0) {
        return GAMECTRL_REGRET;
    }
    if (strcmp(s, "ex") == 0 || strcmp(s, "export") == 0) {
        return GAMECTRL_EXPORT;
    }
    if (isupper(s[0])) return s[0] - 'A';
    if (islower(s[0])) return s[0] - 'a';
    if (!isdigit(s[0])) return -1;
    int tmp = 0, i = 0;
    while (isdigit(s[i])) tmp = tmp * 10 + s[i] - '0', i++;
    return tmp;
}

point_t manual(const game_t game, void* assets)
{
    (void)game, (void)assets;

    point_t pos;
#ifdef DEBUG
    log_i("waiting input. (format: %%d %%d)");
    int x, y;
    prompt(), scanf("%d %d", &x, &y);
    pos.x = y, pos.y = x;
#else
    log_i("waiting input. (format: H 8 or h 8)");
    char input_x[10], input_y[10];
    prompt(), scanf("%s %s", input_x, input_y);
    int first = parse(input_x), second = parse(input_y);
    if (first == GAMECTRL_EXPORT || first == GAMECTRL_REGRET)
        return (point_t){first, second};
    else
        return (point_t){second - 1, first};
#endif
    return pos;
}
