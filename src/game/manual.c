#include "board.h"
#include "game.h"
#include "util.h"

#include <assert.h>
#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

bool same_prefix(char s1[], char s2[], size_t size) {
    int l1 = strlen(s1), l2 = strlen(s2);
    if (l1 < size || l2 < size) return false;
    for (size_t i = 0; i < size; i++) {
        if (s1[i] != s2[i]) return false;
    }
    return true;
}

int parse(char s[], point_t* pos) {
    if (strlen(s) < 2) return 1;
    (*pos).x = (*pos).y = -1;
    if (same_prefix(s, "back", 4)) {
        (*pos).x = GAMECTRL_WITHDRAW;
        if (strlen(s) > 4) {
            (*pos).y = atoi(s + 4);
            return 0;
        } else {
            (*pos).y = 2;
            return 0;
        }
    }
    if (same_prefix(s, "export", 6)) {
        (*pos).x = GAMECTRL_EXPORT;
        return 0;
    }
    if (same_prefix(s, "giveup", 4)) {
        (*pos).x = GAMECTRL_GIVEUP;
        return 0;
    }
    if (same_prefix(s, "switch", 6)) {
        (*pos).x = GAMECTRL_SWITCH_PLAYER;
        if (strlen(s) > 6) {
            (*pos).y = atoi(s + 6);
            return 0;
        } else {
            return 1;
        }
    }
    if (same_prefix(s, "eval", 4)) {
        (*pos).x = GAMECTRL_EVALUATE;
        return 0;
    }

    if (isupper(s[0])) {
        (*pos).y = s[0] - 'A';
    } else if (islower(s[0])) {
        (*pos).y = s[0] - 'a';
    } else {
        return 1;
    }
    (*pos).x = atoi(s + 1) - 1;
    if (!in_board(*pos)) return 1;
    return 0;
}

point_t input_manually(game_t game, const void* assets) {
    (void)game, (void)assets;

#if DEBUG_LEVEL >= 2
    log_i("waiting input. (format: %%d %%d)");
    int x, y;
    prompt_scanf("%d %d", &x, &y);
    return (point_t){x, y};
#else
    log_i("input (eg. \"H8\" / \"h8\" for pos (H, 8), \"back\" for withdraw two moves):");
    char input[10];
    point_t pos = {-1, -1};
    do {
        prompt();
        do {
            fgets(input, sizeof(input), stdin);
            input[strcspn(input, "\n")] = '\0';
        } while (input[0] == '\0');
    } while (parse(input, &pos) && log_e("invalid input."));
    return pos;
#endif
}
