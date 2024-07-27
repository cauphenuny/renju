// author: Cauphenuny <https://cauphenuny.github.io/>
// date: 2024/07/26
#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "util.h"
#include "board.h"
#include "manual.h"
#include "mcts.h"
#include "players.h"

point_t move(int player_type, const board_t board, int id) {
    switch (player_type) {
        case MANUAL: {
            return manual(board);
        }
        case MCTS: {
            mcts_parm_t parm = {.C = 1.414, .M = 0, .TIME_LIMIT = 3000};
            return mcts(board, id, parm);
        }
    }
    return (point_t){-1, -1};
}
