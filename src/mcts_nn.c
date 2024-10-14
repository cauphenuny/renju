// author: Cauphenuny
// date: Tue Sep 24 23:04:29 CST 2024
#include "mcts_nn.h"

#include "board.h"
#include "game.h"
#include "util.h"

#include <stdlib.h>
#include <string.h>

// mcts_nn_parm_t parm;
// 
// typedef struct node_t {
//     struct node_t *parent, *child, *sibling;
//     board_t board;
//     int count;
//     int id;
//     double P;
//     point_t pos;
//     bool expanded;
// } node_t;
// 
// node_t* create_node()
// {
//     node_t* node = (node_t*)malloc(sizeof(node_t));
//     memset(node, 0, sizeof(node_t));
//     return node;
// }
// 
// node_t* update_node(node_t* pre, point_t add_pos)
// {
//     node_t* node = (node_t*)malloc(sizeof(node_t));
//     put(node->board, pre->id, add_pos);
//     node->id = 3 - pre->id;
//     memcpy(node, pre, sizeof(node_t));
//     return node;
// }
// 
// void append_child(node_t* parent, node_t* child)
// {
//     child->sibling = parent->child;
//     child->parent = parent;
//     parent->child = child;
// }
// 
// node_t* select(node_t* node)
// {
//     node_t* best = NULL;
//     double best_score = -1;
//     for (node_t* child = node->child; child != NULL; child = child->sibling) {
//         double score = child->count + child->P;
//         if (score > best_score) {
//             best_score = score;
//             best = child;
//         }
//     }
//     return best;
// }
// 
// void expand(node_t* node)
// {
//     prediction_t prediction = predict(parm.network, node->board, node->id);
//     for (int8_t i = 0; i < BOARD_SIZE; i++) {
//         for (int8_t j = 0; j < BOARD_SIZE; j++) {
//             point_t pos = {i, j};
//             if (node->board[i][j] == 0) {
//                 node_t* child = update_node(node, pos);
//                 child->pos = pos;
//                 child->P = prediction.prob[i][j];
//                 append_child(node, child);
//             }
//         }
//     }
//     node->expanded = true;
// }
// 
// node_t* traverse(node_t* root)
// {
//     if (!root->expanded) {
//         expand(root);
//         return root;
//     }
//     return root;
// }

point_t mcts_nn(game_t game, void* assets)
{
    mcts_nn_param_t param = *((mcts_nn_param_t*)assets);
    log_e("not implemented!");
    return (point_t){-1, -1};
}
