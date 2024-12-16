#include "threat.h"

#include "board.h"
#include "eval.h"
#include "pattern.h"
#include "util.h"
#include "vector.h"

#include <stdlib.h>
#include <string.h>

void test_eval() {
#include "boards.txt"
    vector_t result_5 = vector_new(threat_t, NULL), result_4 = vector_new(threat_t, NULL),
             result_3 = vector_new(threat_t, NULL);
    for (int i = 0; i < 1; i++) {
        board_t board;
        comp_board_t cpboard;
        point_t pos;
        pos = parse_board(board, tests[i].str);
        encode(board, cpboard);

        result_3.size = result_4.size = result_5.size = 0;
        threat_storage_t storage = {[PAT_A3] = &result_3,
                                    [PAT_A4] = &result_4,
                                    [PAT_D4] = &result_4,
                                    [PAT_WIN] = &result_5};

        int tim = record_time();
        for (int t = 0; t < 1; t++) {
            scan_threats(cpboard, 1, storage);
        }
        tim = get_time(tim);
        log("time: %.3lfms", tim * 1.0);
        log("i = %d", i);
        print_emph(board, pos);
        fboard_t mark = {0};
        for_each(threat_t, result_3, threat) {
            point_t point = threat.pos;
            log("got [%s] %d, %d, dir = {%d, %d}", pattern_typename[threat.pattern], point.x,
                point.y, threat.dx, threat.dy);
            mark[point.x][point.y] = 0.05;
        }
        print_prob(board, mark);
        memset(mark, 0, sizeof(mark));
        print_prob(board, mark);
        for_each(threat_t, result_4, threat) {
            point_t point = threat.pos;
            log("got [%s] %d, %d, dir = {%d, %d}", pattern_typename[threat.pattern], point.x,
                point.y, threat.dx, threat.dy);
            mark[point.x][point.y] = 0.2;
        }
        print_prob(board, mark);
        memset(mark, 0, sizeof(mark));
        for_each(threat_t, result_5, threat) {
            point_t point = threat.pos;
            log("got [%s] %d, %d, dir = {%d, %d}", pattern_typename[threat.pattern], point.x,
                point.y, threat.dx, threat.dy);
            mark[point.x][point.y] = 0.4;
        }
        print_prob(board, mark);
    }
    vector_free(&result_3), vector_free(&result_4), vector_free(&result_5);
}

void test_upd() {
#include "boards.txt"
    for (int i = TESTS - 1; i < TESTS; i++) {
        board_t board;
        point_t pos;
        pos = parse_board(board, tests[i].str);
        print_emph(board, pos);
        vector_t array = vector_new(point_t, NULL);
        board[pos.x][pos.y] = 2;
        for_all_dir(d, dx, dy) {
            vector_t tmp = find_relative_points(CONSIST, board, pos, dx, dy);
            print_emph_mutiple(board, tmp);
            vector_cat(array, tmp);
        }
        print_emph_mutiple(board, array);
    }
}

void test_threat() {
#include "boards.txt"
    for (int i = 0; i < 1; i++) {
        board_t board;
        point_t pos;
        pos = parse_board(board, tests[i].str);
        print_emph(board, pos);
        vector_t threat_info = scan_threats_info(board, 1, false);
        for_each(threat_info_t, threat_info, info) {
            log("=========================================");
            log("type: %s", pattern_typename[info.type]);
            log("pos: (%d, %d)", info.action.x, info.action.y);
            print_emph(board, info.action);
            log("consists:");
            print_emph_mutiple(board, info.consists);
            log("defends:");
            print_emph_mutiple(board, info.defenses);
        }
    }
}

void test_threat_tree() {
#include "board_vct.txt"
    for (int i = 0; i < 1; i++) {
        board_t board;
        parse_board(board, tests[i].str);
        // vector_t forest = get_threat_forest(board, 1, true);
        // for_each_ptr(threat_tree_node_t*, forest, proot) {
        //     threat_tree_node_t* root = *proot;
        //     print_threat_tree(root);
        // }
        vector_t vcf_sequence = vcf(board, 1);
        print_vcf(vcf_sequence);
        vector_free(&vcf_sequence);
    }
}