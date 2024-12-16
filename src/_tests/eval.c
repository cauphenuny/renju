#include "eval.h"

#include "board.h"
#include "pattern.h"
#include "util.h"
#include "vector.h"

#include <stdlib.h>
#include <string.h>

void test_eval() {
#include "boards.txt"
    // vector_t result_5 = vector_new(point_t), result_4 = vector_new(point_t), result_3 =
    // vector_new(point_t); for (int i = TESTS - 1; i < TESTS; i++) {
    //     board_t board;
    //     comp_board_t cpboard;
    //     point_t pos;
    //     pos = parse_board(board, tests[i].str);
    //     encode(board, cpboard);

    //     result_3.size = result_4.size = result_5.size = 0;
    //     point_storage_t storage = {[PAT_A2] = &result_3,
    //                                [PAT_A3] = &result_4,
    //                                [PAT_D3] = NULL,  // FIXME: add D3 will change A2 behavior
    //                                [PAT_A4] = &result_5,
    //                                [PAT_D4] = &result_5};

    //     int tim = record_time();
    //     for (int t = 0; t < 1; t++) {
    //         find_points(cpboard, 2, storage);
    //     }
    //     tim = get_time(tim);
    //     log("time: %.3lfms", tim * 1.0);
    //     log("i = %d", i);
    //     print_emph(board, pos);
    //     fboard_t mark = {0};
    //     for (int j = 0; j < result_3.size; j++) {
    //         log("got 3-point %d, %d", result_3.points[j].x, result_3.points[j].y);
    //         mark[result_3.points[j].x][result_3.points[j].y] = 0.05;
    //     }
    //     print_prob(board, mark);
    //     memset(mark, 0, sizeof(mark));
    //     print_prob(board, mark);
    //     for (int j = 0; j < result_4.size; j++) {
    //         log("got 4-point %d, %d", result_4.points[j].x, result_4.points[j].y);
    //         mark[result_4.points[j].x][result_4.points[j].y] = 0.2;
    //         // print_emph(board, result_4.points[j]);
    //     }
    //     print_prob(board, mark);
    //     memset(mark, 0, sizeof(mark));
    //     for (int j = 0; j < result_5.size; j++) {
    //         log("got 5-point %d, %d", result_5.points[j].x, result_5.points[j].y);
    //         mark[result_5.points[j].x][result_5.points[j].y] = 0.6;
    //         // print_emph(board, result_5.points[j]);
    //     }
    //     print_prob(board, mark);
    // }
    // free(result_5.points);
    // free(result_4.points);
}

void test_upd() {
#include "boards.txt"
    for (int i = TESTS - 1; i < TESTS; i++) {
        board_t board;
        point_t pos;
        pos = parse_board(board, tests[i].str);
        print_emph(board, pos);
        vector_t array = vector_new(point_t);
        board[pos.x][pos.y] = 2;
        for_all_dir(d, dx, dy) {
            vector_t tmp = find_relative_points(CONSIST, board, pos, dx, dy);
            print_emph_mutiple(board, tmp);
            vector_cat(array, tmp);
        }
        print_emph_mutiple(board, array);
    }
}