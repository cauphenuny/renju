// author: Cauphenuny
// date: 2024/07/27
#include "board.h"
#include "game.h"
#include "init.h"
#include "players.h"
#include "util.h"

#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

game_t start_game(int player1, int player2, int first_player)
{
    log("start game: %s vs %s, first player: %s", player_name[player1], player_name[player2],
        player_name[first_player]);
    game_t game = new_game(first_player);
    point_t pos;
    game_print(game);
    while (1) {
        int id = game.cur_player;
        int tim;
        if (id == 1) {
            log_i("------ step " L_GREEN "#%d" NONE ", player1's turn ------", game.count + 1);
            tim = record_time(), pos = move(player1, NULL, game);
        } else {
            log_i("------ step " L_RED "#%d" NONE ", player2's turn ------", game.count + 1);
            tim = record_time(), pos = move(player2, NULL, game);
        }
        log_i("time: %dms, chose " BOLD UNDERLINE "(%c, %d)" NONE, get_time(tim), pos.y + 'A',
              pos.x + 1);

        if (!available(game.board, pos)) {
            log_e("invalid position!");
            // game_export(game, "game->");
            // prompt_getch();
            continue;
        }
        extern int _is_banned_enable_log;
        _is_banned_enable_log = 1;
        int ban = is_banned(game.board, pos, id);
        _is_banned_enable_log = 0;
        if (game.cur_player == game.first_player) {
            if (ban) {
                log_e("banned position! (%s)", pattern4_typename[ban]);
                prompt();
                char c = 0;
                while (c != 'q' && c != 'c') c = getchar();
                if (c == 'c') continue;
                else {
                    game.winner = 3 - id;
                    return game;
                }
            }
        }

        game_add_step(&game, pos);
        game_print(game);

        if (check_draw(game.board)) {
            log_i("draw.");
            game.winner = 0;
            return game;
        }
        if (check(game.board, pos)) {
            log_i("player%d wins.", id);
            game.winner = id;
            return game;
        }
    }
    return game;
}

int results[5];

void signal_handler(int signum)
{
    log_s("\n");
    log("received signal %d, terminate.", signum);
    int r1 = results[1], r2 = results[2];
    if (r1 + r2) {
        log_i("results: p1/p2/draw: %d/%d/%d (%.2lf%%), 1st/2nd: %d/%d (%.2lf%%)", r1, r2,
              results[0], (double)r1 / (r1 + r2) * 100, results[3], results[4],
              (double)results[3] / (r1 + r2) * 100);
    }
    exit(0);
}

// void load_preset(game_t* game) {
// //clang-format off
// //clang-format on
// }

#define PRESET_SIZE 3
const int presets[PRESET_SIZE][2] = {{MCTS, MANUAL}, {MANUAL, MCTS}, {MCTS, MCTS}};

int main(void)
{
    signal(SIGINT, signal_handler);

    log_i("gomoku v%s", VERSION);

    init();

    log_i("available modes: ");
    for (int i = 0; i < PRESET_SIZE; i++) {
        log_i("#%d: %s vs %s", i + 1, player_name[presets[i][0]], player_name[presets[i][1]]);
    }
    log_i("#0: custom");

    int mode = -1;
    do {
        prompt(), scanf("%d", &mode);
    } while (mode < 0 || mode > PRESET_SIZE);

    int player1, player2;
    if (!mode) {
        log_i("available players:");
        for (int i = 0; i < PLAYER_CNT; i++) log_i("#%d: %s", i, player_name[i]);
        log_i("input P1 P2:");
        do prompt(), scanf("%d%d", &player1, &player2);
        while (player1 < 0 || player1 >= PLAYER_CNT || player2 < 0 || player2 >= PLAYER_CNT);
    } else {
        player1 = presets[mode - 1][0], player2 = presets[mode - 1][1];
    }

    int id = 1;
    while (1) {
        game_t game = start_game(player1, player2, id);
        results[game.winner]++;
        if (game.winner == id) {
            results[3]++;
        } else if (game.winner == 3 - id) {
            results[4]++;
        }
        log_i("results: p1/p2/draw: %d/%d/%d, 1st/2nd: %d/%d", results[1], results[2], results[0],
              results[3], results[4]);
        // #ifdef DEBUG
        //         char ch[3];
        //         do {
        //             prompt();
        //             int step;
        //             char identifier[64];
        //             scanf("%s", ch);
        //             switch (ch[0]) {
        //             case 'p':
        //                 scanf("%d%s", &step, identifier);
        //                 for (int i = 0, x, y; i < step; i++) {
        //                     x = game.steps[i].x;
        //                     y = game.steps[i].y;
        //                     printf("%s.board[%d][%d]=%d;"
        //                            "%s.steps[%s.steps_cnt++]=(point_t){%d, %d};",
        //                            identifier, x, y, game.board[x][y], identifier,
        //                            identifier, x, y);
        //                 }
        //                 printf("\n");
        //                 break;
        //             case 'r':
        //                 scanf("%d", &step);
        //                 memset(&game.board, 0, sizeof(game.board));
        //                 int cur_id = game.first_id;
        //                 for (int i = 0, x, y; i < step; i++) {
        //                     x = game.steps[i].x;
        //                     y = game.steps[i].y;
        //                     game.board[x][y] = cur_id;
        //                     cur_id = 3 - cur_id;
        //                 }
        //                 game.current_id = 3 - cur_id;
        //                 goto start_game;
        //             }
        //         } while (ch[0] != 'n');
        // #endif
        id = 3 - id;
    }
    return 0;
}
