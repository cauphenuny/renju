// author: Cauphenuny <https://cauphenuny.github.io/>
// date: 2024/07/24
#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "util.h"
#include "board.h"
#include "players.h"

/*
   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | A |
---+---+---+---+---+---+---+---+---+---+---+---+
 0 |   |   |   |   |   |   |   |   |   |   |   |
---+---+---+---+---+---+---+---+---+---+---+---+
 1 |   |   |   |   |   |   |   |   |   |   |   |
---+---+---+---+---+---+---+---+---+---+---+---+
 2 |   |   |   |   |   |   |   |   |   |   |   |
---+---+---+---+---+---+---+---+---+---+---+---+
 3 |   |   |   |   |   |   |   |   |   |   |   |
---+---+---+---+---+---+---+---+---+---+---+---+
 4 |   |   |   |   | x |   |   |   |   |   |   |
---+---+---+---+---+---+---+---+---+---+---+---+
 5 |   |   |   | x | o | o |   | o | x |   |   |
---+---+---+---+---+---+---+---+---+---+---+---+
 6 |   |   |   |   | x |   | o |   |   |   |   |
---+---+---+---+---+---+---+---+---+---+---+---+
 7 |   |   |   |   |   | o |   | o |   |   |   |
---+---+---+---+---+---+---+---+---+---+---+---+
 8 |   |   |   |   |   |   |   |   | x |   |   |
---+---+---+---+---+---+---+---+---+---+---+---+
 9 |   |   |   |   |   |   |   |   |   |   |   |
---+---+---+---+---+---+---+---+---+---+---+---+
 A |   |   |   |   |   |   |   |   |   |   |   |
---+---+---+---+---+---+---+---+---+---+---+---+
reverse_flag = 1;
board[4][4] = board[5][3] = board[6][4] = board[5][8] = board[8][8] = 2;
board[5][4] = board[5][5] = board[7][5] = board[5][7] = board[6][6] = board[7][7] = 1;
*/

int game(int player1, int player2, int first) {
    board_t board;
    memset(board, 0, sizeof(board));
    print(board);
    point_t pos;
    int players[2] = {player1, player2};
    int id = first;
    while (1) {
        log("player%d's turn.", id);
        while (put(board, id, pos = move(players[id - 1], board, id))) {
            loge("invalid position!");
            getchar();
        }
        print(board);
        refresh(board);
        if (check_draw(board)) { log("draw."); return 0; }
        if (check(board, pos)) { log("player%d wins.", id); return id; }
        id = 3 - id;
    }
}
