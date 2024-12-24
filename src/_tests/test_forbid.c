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
            case 'o': dest[x][y++] = 1; break;
            case 'x': dest[x][y++] = 2; break;
            case '?': pos = (point_t){x, y++}; break;
            case '\\': y = 0, x++;
        }
    }
    for (int line = 0; line < BOARD_SIZE / 2; line++) {
        int match_line = BOARD_SIZE - 1 - line;
        for (int col = 0; col < BOARD_SIZE; col++) {
            int tmp = dest[line][col];
            dest[line][col] = dest[match_line][col];
            dest[match_line][col] = tmp;
        }
    }
    pos.x = BOARD_SIZE - 1 - pos.x;
    return pos;
}

extern bool enable_forbid_log;

int test_forbid(void)
{
    enable_forbid_log = true;
#include "boards.txt"
    board_t board;
    point_t pos;
    for (int i = 0; i < TESTS; i++) {
        log_l("i = %d", i);
        pos = parse_board(board, tests[i].str);
        print_emph(board, pos);
        const pattern4_t forbid = is_forbidden(board, pos, 1, -1);
        log_l("got %s, expected %s", pattern4_typename[forbid], pattern4_typename[tests[i].id]);
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