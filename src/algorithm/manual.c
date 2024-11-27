// author: Cauphenuny
// date: 2024/07/26

#include "board.h"
#include "game.h"
#include "util.h"

#include <assert.h>
#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int parse(char s[])
{
    if (strcmp(s, "back") == 0 || strcmp(s, "withdraw") == 0) {
        return GAMECTRL_WITHDRAW;
    }
    if (strcmp(s, "export") == 0) {
        return GAMECTRL_EXPORT;
    }
    if (strcmp(s, "give") == 0) {  // give up
        return GAMECTRL_GIVEUP;
    }
    if (strcmp(s, "change") == 0) {
        return GAMECTRL_CHANGE_PLAYER;
    }
    if (isupper(s[0])) return strlen(s) == 1 ? s[0] - 'A' : -1;
    if (islower(s[0])) return strlen(s) == 1 ? s[0] - 'a' : -1;
    if (!isdigit(s[0]) && s[0] != '-') return -1;
    int tmp = 0, i = 0;
    while (isdigit(s[i])) tmp = tmp * 10 + s[i] - '0', i++;
    return tmp * (s[0] == '-' ? -1 : 1);
}

point_t manual(const game_t game, const void* assets)
{
    (void)game, (void)assets;

#if DEBUG_LEVEL >= 2
    log_i("waiting input. (format: %%d %%d)");
    int x, y;
    prompt_scanf("%d %d", &x, &y);
    return (point_t){x, y};
#else
    log_i("input (eg. \"H 8\" / \"h 8\" for pos (H, 8), \"back 2\" for withdraw two moves):");
    char input_x[10], input_y[10];
    prompt_scanf("%s %s", input_x, input_y);
    const int first = parse(input_x), second = parse(input_y);
    switch (first) {
        case GAMECTRL_GIVEUP:
        case GAMECTRL_EXPORT:
        case GAMECTRL_WITHDRAW:
        case GAMECTRL_CHANGE_PLAYER: return (point_t){first, second};
        default: {
            const point_t pos = (point_t){second - 1, first};
            if (in_board(pos)) return pos;
            log_e("invalid input.");
            return manual(game, assets);
        }
    }
#endif
}
