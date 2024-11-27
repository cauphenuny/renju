#include "board.h"
#include "util.h"
#include "pattern.h"

#include <stdlib.h>
#include <string.h>

point_t parse_board(board_t dest, const char* str)
{
    memset(dest, 0, sizeof(board_t));
    int x = 0, y = 0, l = strlen(str);
    point_t pos;
    for (int i = 0; i < l; i++) {
        switch (str[i]) {
            case '.': y++; break;
            case 'o': dest[x][++y] = 1; break;
            case 'x': dest[x][++y] = 2; break;
            case '?': pos = (point_t){x, ++y}; break;
            case '\\': y = 0, x++;
        }
    }
    return pos;
}

int test_forbid(void)
{
#include "boards.txt"
    board_t board;
    point_t pos;
    for (int i = 0; i < TESTS; i++) {
        log("i = %d", i);
        pos = parse_board(board, tests[i].str);
        emphasis_print(board, pos);
        const pattern4_t forbid = is_forbidden(board, pos, 1, true);
        log("got %s, expected %s", pattern4_typename[forbid], pattern4_typename[tests[i].id]);
        if (forbid != tests[i].id) {
            log_e("failed.");
            return 1;
        }
    }
    return 0;
}

/*
"


*/