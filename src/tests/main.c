#include "board.h"
#include "game.h"
#include "init.h"
#include "minimax.h"
#include "server.h"
#include "util.h"

#include <assert.h>

#ifndef TEST
#    error "define TEST to run unit tests!"
#endif

int test_forbid(void);

static int test_pattern(void)
{
    /*
 2538 [. o . o o o . . .]: level = dead 4
 3562 [. o o x o x x x o]: level = empty
18792 [x x o x o . . . .]: level = dead 1
 3294 [. o o o o x . . .]: level = dead 4
    */
    for (int idx = 0; idx < PATTERN_SIZE; idx++) {
        const segment_t seg = segment_decode(idx);
        if (seg.pieces[WIN_LENGTH - 1] != SELF_PIECE) continue;
        segment_print(segment_decode(idx));
    }
    return 0;
}

void test_minimax(void)
{
    const game_t game = game_import(
        300, 1, 219,
        (point_t[]){
            {7, 7},  {6, 7},   {7, 8},   {7, 9},   {6, 8},   {8, 8},   {8, 6},   {9, 5},   {9, 7},
            {5, 9},  {10, 8},  {7, 5},   {12, 6},  {11, 9},  {11, 7},  {9, 9},   {10, 7},  {8, 7},
            {10, 9}, {10, 6},  {11, 6},  {12, 7},  {9, 8},   {12, 5},  {11, 5},  {11, 4},  {10, 4},
            {9, 3},  {13, 4},  {13, 5},  {7, 10},  {8, 9},   {6, 9},   {10, 10}, {5, 8},   {8, 11},
            {8, 10}, {4, 8},   {6, 10},  {5, 10},  {10, 1},  {6, 11},  {11, 2},  {12, 3},  {10, 3},
            {10, 2}, {9, 4},   {12, 1},  {11, 0},  {4, 7},   {9, 6},   {8, 5},   {4, 6},   {13, 7},
            {4, 4},  {8, 3},   {14, 6},  {13, 6},  {5, 5},   {6, 6},   {8, 4},   {6, 4},   {8, 1},
            {11, 1}, {12, 0},  {10, 0},  {4, 11},  {11, 8},  {5, 13},  {5, 7},   {3, 9},   {9, 2},
            {3, 11}, {3, 10},  {4, 12},  {2, 10},  {4, 13},  {4, 14},  {1, 10},  {6, 14},  {3, 12},
            {2, 11}, {3, 13},  {2, 13},  {5, 11},  {2, 14},  {2, 12},  {5, 12},  {5, 3},   {4, 10},
            {3, 5},  {6, 2},   {14, 8},  {1, 12},  {14, 7},  {14, 5},  {5, 4},   {5, 2},   {12, 8},
            {2, 6},  {0, 4},   {7, 13},  {7, 2},   {6, 3},   {14, 10}, {14, 9},  {2, 5},   {4, 5},
            {4, 3},  {13, 9},  {13, 8},  {3, 14},  {5, 14},  {3, 4},   {3, 0},   {3, 2},   {3, 7},
            {1, 9},  {0, 10},  {1, 11},  {6, 5},   {3, 6},   {4, 0},   {4, 1},   {7, 4},   {1, 3},
            {5, 0},  {6, 0},   {6, 1},   {7, 1},   {0, 3},   {2, 0},   {0, 2},   {0, 5},   {9, 11},
            {0, 1},  {10, 13}, {11, 10}, {7, 12},  {1, 2},   {2, 3},   {9, 0},   {12, 12}, {11, 13},
            {0, 11}, {0, 9},   {11, 12}, {10, 12}, {4, 2},   {9, 14},  {13, 14}, {12, 13}, {1, 5},
            {2, 4},  {7, 14},  {2, 9},   {2, 7},   {12, 9},  {12, 4},  {13, 11}, {10, 11}, {1, 4},
            {3, 1},  {2, 2},   {14, 2},  {13, 3},  {12, 10}, {13, 1},  {12, 11}, {14, 0},  {1, 6},
            {0, 7},  {7, 3},   {13, 12}, {11, 3},  {6, 13},  {12, 14}, {14, 14}, {3, 8},   {8, 13},
            {9, 13}, {10, 14}, {8, 12},  {13, 2},  {2, 1},   {0, 12},  {13, 0},  {1, 13},  {7, 11},
            {14, 4}, {12, 2},  {13, 10}, {13, 13}, {14, 12}, {5, 1},   {14, 13}, {14, 3},  {14, 1},
            {10, 5}, {0, 6},   {0, 8},   {1, 8},   {8, 14},  {8, 0},   {7, 0},   {11, 14}, {1, 1},
            {1, 0},  {4, 9},   {0, 0},   {6, 12},  {1, 7},   {7, 6},   {0, 13},  {3, 3},   {0, 14},
            {1, 14}, {14, 11}, {11, 11}});
    const point_t pos = minimax(game, NULL);
    emphasis_print(game.board, pos);
    log("got pos %d, %d", pos.x, pos.y);
    assert(in_board(pos));
}

int main()
{
    init();

    int ret = 0;
    log("running test");

    log("test pattern");
    test_pattern();

    log("test forbid");
    ret = test_forbid();
    if (ret) return ret;

    log_i("forbid tests passed.");

    void mcts_test_entrance(void);
    mcts_test_entrance();

    log_i("mcts tests passed.");

    test_minimax();

    log_i("minimax tests passed.");

    log("run zobrist test? [y/n]");

    const char ch = prompt_pause();

    if (ch == 'y') {
        start_game(preset_players[MINIMAX], preset_players[MINIMAX], 1, 2000);
        log("zobrist tests passed.");
    }


    log_i("all tests passed.");
    return 0;
}