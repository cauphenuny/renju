#include "board.h"
#include "eval.h"
#include "game.h"
#include "init.h"
#include "manual.h"
#include "minimax.h"
#include "pattern.h"
#include "players.h"
#include "util.h"
#include "vct.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#ifndef TEST
#error "define TEST to run unit tests!"
#endif

int test_forbid(void);

int test_pattern(void) {
    /*
 2538 [. o . o o o . . .]: level = dead 4
 3562 [. o o x o x x x o]: level = empty
18792 [x x o x o . . . .]: level = dead 1
 3294 [. o o o o x . . .]: level = dead 4
    */
    for (int idx = 0; idx < SEGMENT_SIZE; idx++) {
        const segment_t seg = decode_segment(idx, PIECE_SIZE);
        if (!segment_valid(seg)) continue;
        print_segment(seg, true);
    }
    return 0;
}

void test_one_step(game_t* game) {
    print_game((*game));
    char buffer[1024];
    board_serialize((*game).board, buffer);
    printf("%s", buffer);
    point_t pos = minimax((*game), preset_players[MINIMAX_ADV].assets);
    print_emph((*game).board, pos);
    log("got pos %d, %d: %c%d", pos.x, pos.y, READABLE_POS(pos));
    assert(in_board(pos));
    add_step(game, pos);
    print_game((*game));
    serialize_game((*game), "");
    vector_t vct_seq = vct(false, (*game).board, (*game).cur_id, 50);
    log("vct: %d", vct_seq.size);
    vector_free(vct_seq);
    pos = input_manually((*game), NULL);
    add_step(game, pos);
    serialize_game((*game), "");
}

void test_minimax(void) {
    // clang-format off
    // game_t game = restore_game(5000,7,(point_t[]){{7,7},{8,6},{7,5},{7,6},{6,6},{5,5},{5,7}});
    game_t game[4];
    game[0] = 
restore_game(10000,21,(point_t[]){{7,7},{6,7},{8,6},{7,8},{8,9},{8,5},{9,9},{5,8},{7,6},{6,8},{8,8},{6,6},{6,5},{6,9},{6,10},{4,8},{3,8},{8,7},{5,10},{3,9},{5,7}});
    game[1] = 
restore_game(15000,5,(point_t[]){{7,7},{6,6},{7,5},{6,7},{6,5}});
    game[2] = 
restore_game(30000,28,(point_t[]){{7,7},{7,8},{6,6},{8,8},{6,8},{8,6},{6,7},{6,5},{8,7},{5,7},{5,9},{4,10},{6,10},{6,9},{5,8},{9,7},{3,8},{4,8},{4,9},{3,10},{5,10},{6,11},{7,6},{8,5},{7,5},{7,4},{2,7},{1,6},{5,6},{9,6},{10,7},{9,8},{9,5}});
    game[3] = 
restore_game(10000,27,(point_t[]){{7,7},{6,7},{8,6},{7,8},{8,9},{8,5},{9,9},{5,8},{7,6},{6,8},{8,8},{6,6},{6,5},{4,8},{3,8},{6,9},{6,10},{8,7},{5,10},{3,9},{5,7},{4,9},{7,9},{9,6},{10,5},{2,9},{5,9}});
    //restore_game(15000,28,(point_t[]){{7,7},{7,9},{6,8},{8,6},{8,8},{6,6},{5,6},{7,8},{5,7},{6,7},{5,8},{5,9},{6,9},{6,4},{4,7},{6,5},{6,3},{3,6},{5,3},{8,5},{4,3},{7,3},{3,3},{2,3},{8,11},{7,10},{8,10},{8,12}});
    // clang-format on
    for (size_t i = 0; i < sizeof(game) / sizeof(game_t); i++) {
        test_one_step(&game[i]);
        test_one_step(&game[i]);
    }
}

void test_upd();
void test_eval();
void test_threat();
void test_threat_tree();
void test_threat_seq();

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

int main(int argc, char** argv) {
    init();
    int ret = 0;
    log("running test");

    char s[64] = {0};
    if (argc < 2) {
        log("input test name: ");
        prompt_scanf("%s", s);
        argv[1] = s;
    }
    bool all = 0;
    if (strcmp(argv[1], "all") == 0) all = 1;

    if (strcmp(argv[1], "vector") == 0 || all) {
        log("test vector");
        test_vector();
    }
    if (strcmp(argv[1], "pattern") == 0 || all) {
        log("test pattern");
        test_pattern();
    }
    if (strcmp(argv[1], "forbid") == 0 || all) {
        log("test forbid");
        ret = test_forbid();
        if (ret) return ret;
        log_i("forbid tests passed.");
    }
    if (strcmp(argv[1], "minimax") == 0 || all) {
        log("test minimax");
        test_minimax();
        log_i("minimax tests passed.");
    }
    if (strcmp(argv[1], "mcts") == 0 || all) {
        void mcts_test_entrance(void);
        log("test mcts");
        mcts_test_entrance();
        log_i("mcts tests passed.");
    }
    if (strcmp(argv[1], "threat") == 0 || all) {
        log("test threat");
        test_threat();
    }
    if (strcmp(argv[1], "threat_seq") == 0 || all) {
        log("test threat_seq");
        test_threat_seq();
    }
    if (strcmp(argv[1], "threat_tree") == 0 || all) {
        log("test threat_tree");
        test_threat_tree();
    }

    log_i("test `%s` passed.", argv[1]);
    return 0;
}