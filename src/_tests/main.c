#include "board.h"
#include "game.h"
#include "init.h"
#include "minimax.h"
#include "pattern.h"
#include "util.h"

#include <assert.h>
#include <string.h>

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
    for (int idx = 0; idx < SEGMENT_MASK; idx++) {
        const segment_t seg = decode_segment(idx);
        if (!segment_valid(seg)) continue;
        print_segment(seg, false);
    }
    return 0;
}

void test_minimax(void)
{
    game_t game = restore_game(
        2000, 90,
        (point_t[]){
            {7, 7},   {8, 6},   {7, 8},  {7, 6},  {6, 6},  {5, 5},   {6, 7},   {8, 7},  {6, 8},
            {6, 5},   {6, 9},   {6, 10}, {9, 8},  {8, 8},  {8, 5},   {8, 9},   {8, 10}, {7, 9},
            {9, 7},   {5, 4},   {4, 3},  {5, 11}, {4, 12}, {9, 9},   {5, 6},   {4, 5},  {3, 5},
            {3, 6},   {6, 3},   {4, 11}, {6, 11}, {10, 9}, {11, 9},  {11, 11}, {2, 7},  {7, 4},
            {6, 4},   {8, 4},   {7, 3},  {5, 3},  {10, 6}, {2, 5},   {1, 4},   {1, 5},  {8, 3},
            {9, 3},   {10, 4},  {7, 1},  {8, 0},  {11, 3}, {12, 3},  {3, 12},  {2, 11}, {1, 9},
            {3, 7},   {13, 5},  {10, 2}, {12, 5}, {11, 5}, {9, 4},   {9, 2},   {11, 8}, {12, 7},
            {12, 11}, {10, 11}, {12, 9}, {5, 7},  {4, 8},  {12, 12}, {4, 7},   {4, 9},  {13, 1},
            {13, 10}, {10, 12}, {9, 13}, {4, 10}, {1, 7},  {5, 1},   {5, 2},   {3, 2},  {5, 10},
            {3, 8},   {2, 8},   {13, 6}, {13, 4}, {7, 13}, {6, 12},  {13, 9},  {10, 5}, {10, 3}});
    const point_t pos = minimax(game, NULL);
    print_emph(game.board, pos);
    log("got pos %d, %d", pos.x, pos.y);
    assert(in_board(pos));
}

void test_upd();

int main()
{
    init();
    int ret = 0;
    log("running test");

    // log("test pattern");
    // test_pattern();

    // log("test forbid");
    // ret = test_forbid();
    // if (ret) return ret;

    // log_i("forbid tests passed.");

    // void mcts_test_entrance(void);
    // mcts_test_entrance();

    // log_i("mcts tests passed.");

    // test_minimax();

    // log_i("minimax tests passed.");

    // log("run zobrist test? [y/n]");

    // const int ch = prompt_pause();

    // if (ch == 'y') {
    //     start_game(preset_players[MINIMAX], preset_players[MINIMAX], 1, 2000, NULL);
    //     log("zobrist tests passed.");
    // }

    // void test_neuro();
    // log("test neuro");
    // test_neuro();

    log("test vcx");
    test_upd();
    log_i("all tests passed.");
    return 0;
}