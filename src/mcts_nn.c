// author: Cauphenuny <https://cauphenuny.github.io/>
// date: 2024/09/22
#include "mcts_nn.h"

#include "board.h"
#include "game.h"

typedef struct node_t {
    struct pmnode_t *parent, *child, *sibling;
    board_t board;
    int count;
    double P;
    point_t pos;
    bool expanded;
} node_t;

point_t mcts_nn(game_t game, mcts_nn_parm_t parm)
{
    return (point_t){0, 0};
}
