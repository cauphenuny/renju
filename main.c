// author: Cauphenuny <https://cauphenuny.github.io/>
// date: 2024/07/27
#include "game.h"
#include "mcts.h"
#include "players.h"
#include "util.h"
#include "zobrist.h"

#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void export(game_t game, int step)
{
    int id = game.first_id;
    printf("game->first_id=%d;", id);
    for (int i = 0, x, y; i < step; i++) {
        x = game.steps[i].x;
        y = game.steps[i].y;
        printf("game->board[%d][%d]=%d;"
               "game->steps[game->step_cnt++]=(point_t){%d, %d};",
               x, y, game.board[x][y], x, y);
        id = 3 - id;
    }
    printf("game->current_id=%d;", id);
}

game_t start_game(game_t game)
{
    point_t pos;
    int id = game.current_id;
    print(game.board);
    while (1) {
        if (id == 1)
            log("step " L_GREEN "#%d" NONE ", player1's turn.",
                game.step_cnt + 1);
        else
            log("step " L_RED "#%d" NONE ", player2's turn.",
                game.step_cnt + 1);
        game.current_id = id;
        pos = move(game.players[id], game);
        if (!available(game.board, pos)) {
            log_e("invalid position!");
            export(game, game.step_cnt);
            // prompt_getch();
            continue;
        }
        if (id == game.first_id) {
            // int ban = banned(game.board, pos, id);
            int ban = POS_ACCEPT;
            if (ban != POS_ACCEPT) {
                log_e("banned position! (%s)", POS_BAN_MSG[ban]);
                export(game, game.step_cnt);
                prompt_getch();
                continue;
            }
        }
        game.steps[game.step_cnt++] = pos;
        put(game.board, id, pos);
        print(game.board);
        refresh(game.board);
        if (check_draw(game.board)) {
            log("draw.");
            game.winner_id = 0;
            return game;
        }
        if (check(game.board, pos)) {
            log("player%d wins.", id);
            game.winner_id = id;
            return game;
        }
        id = 3 - id;
    }
    return game;
}

int results[5];

void signal_handler(int signum) {
    log_s("\n");
    log("received signal %d, terminate.", signum);
    int r1 = results[1], r2 = results[2];
    if (r1 + r2) {
        log_i(
            "results: p1/p2/draw: %d/%d/%d (%.2lf%%), 1st/2nd: %d/%d (%.2lf%%)",
            r1, r2, results[0], (double)r1 / (r1 + r2) * 100, results[3],
            results[4], (double)results[3] / (r1 + r2) * 100);
    }
    exit(0);
}

// void load_preset(game_t* game) {
// //clang-format off
// //clang-format on
// }

int main(void) {
    signal(SIGINT, signal_handler);
    // test_ban();
    log("gomoku v%s", VERSION);
    int id = 1;
    while (1) {
        game_t game;
        memset(&game, 0, sizeof(game_t));
        game.first_id = id;
        game.current_id = id;
        // load_preset(&game);
        game.players[1] = MCTS;
        game.players[2] = MANUAL;
        zobrist_init();
        players_init();
#ifdef DEBUG
    start_game:
#endif
        game = start_game(game);
        results[game.winner_id]++;
        if (game.winner_id == id) {
            results[3]++;
        } else if (game.winner_id == 3 - id) {
            results[4]++;
        }
        log_i("results: p1/p2/draw: %d/%d/%d, 1st/2nd: %d/%d", results[1],
              results[2], results[0], results[3], results[4]);
#ifdef DEBUG
        char ch[3];
        do {
            prompt();
            int step;
            char identifier[64];
            scanf("%s", ch);
            switch (ch[0]) {
            case 'p':
                scanf("%d%s", &step, identifier);
                for (int i = 0, x, y; i < step; i++) {
                    x = game.steps[i].x;
                    y = game.steps[i].y;
                    printf("%s.board[%d][%d]=%d;"
                           "%s.steps[%s.steps_cnt++]=(point_t){%d, %d};",
                           identifier, x, y, game.board[x][y], identifier,
                           identifier, x, y);
                }
                printf("\n");
                break;
            case 'r':
                scanf("%d", &step);
                memset(&game.board, 0, sizeof(game.board));
                int cur_id = game.first_id;
                for (int i = 0, x, y; i < step; i++) {
                    x = game.steps[i].x;
                    y = game.steps[i].y;
                    game.board[x][y] = cur_id;
                    cur_id = 3 - cur_id;
                }
                game.current_id = 3 - cur_id;
                goto start_game;
            }
        } while (ch[0] != 'n');
#endif
        id = 3 - id;
    }
    return 0;
}
