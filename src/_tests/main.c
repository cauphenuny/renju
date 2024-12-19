#include "board.h"
#include "eval.h"
#include "game.h"
#include "init.h"
#include "manual.h"
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
        print_segment(seg, true);
    }
    return 0;
}

void test_minimax(void)
{
    // clang-format off
    // game_t game = restore_game(5000,7,(point_t[]){{7,7},{8,6},{7,5},{7,6},{6,6},{5,5},{5,7}});
    game_t game = 
    restore_game(2000,24,(point_t[]){{7,7},{4,8},{6,6},{8,8},{7,5},{7,9},{7,6},{7,8},{5,6},{8,6},{7,4},{7,3},{6,5},{4,7},{6,3},{5,8},{6,8},{8,7},{6,9},{6,7},{4,6},{3,6},{8,3},{9,2}});
    // clang-format on
    bool true_value = true;
    while (1) {
        {
            const point_t pos = minimax(game, &true_value);
            print_emph(game.board, pos);
            log("got pos %d, %d", pos.x, pos.y);
            assert(in_board(pos));
        }
        point_t pos = input_manually(game, NULL);
        add_step(&game, pos);
        print_game(game);
        log("eval: %lld", eval(game.board));
        game = backward(game, game.count - 1);
    }
}

void test_upd();
void test_eval();
void test_threat();
void test_threat_tree();

void test_vector() {
    vector_t vec = vector_new(int, NULL);
    for (int i = 0; i < 10; i++) {
        vector_push_back(vec, i);
    }
    for_each(int, vec, x) { log("%d", x); }
    vector_free(vec);

    vector_t str = vector_new(char, NULL);
    char s[] = "Hello World";
    for (int i = 0, l = strlen(s); i < l; i++) {
        vector_push_back(str, s[i]);
    }
    char tmp = '\0';
    vector_t str2 = vector_new(char, NULL);
    tmp = '\n', vector_push_back(str2, tmp);
    tmp = '\0', vector_push_back(str2, tmp);
    vector_cat(str, str2);
    printf("%s", (char*)str.data);
}

int main()
{
    init();
    int ret = 0;
    log("running test");

    // log("test vector");
    // test_vector();

    // log("test pattern");
    // test_pattern();

    // log("test forbid");
    // ret = test_forbid();
    // if (ret) return ret;

    // log_i("forbid tests passed.");

    // void mcts_test_entrance(void);
    // mcts_test_entrance();

    // log_i("mcts tests passed.");

    log("test minimax");
    test_minimax();

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

    // log("test threat");
    // test_upd();
    // test_eval();
    // test_threat();
    test_threat_tree();

    log_i("all tests passed.");
    return 0;
}