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

static game_t start_game(player_t p1, player_t p2, int first_id, int time_limit)
{
    player_t players[] = {{}, p1, p2};
    const char* colors[] = {"", L_GREEN, L_RED};
    log("start game: %s vs %s, first player: %d", p1.name, p2.name, first_id);
    game_t game = new_game(first_id, time_limit);
    point_t pos;
    game_print(game);
    while (1) {
        int id = game.cur_id;
        int tim;
        log_i("------ step %s#%d" NONE ", player1's turn ------", colors[id], game.count + 1);
        tim = record_time();
        pos = players[id].move(game, players[id].assets);

        if (pos.x == GAMECTRL_REGRET) {
            if (pos.y > 0 && pos.y <= game.count) {
                game = game_backward(game, game.count - pos.y);
                game_print(game);
            } else log("invalid input");
            continue;
        }
        if (pos.x == GAMECTRL_EXPORT) {
            if (pos.y > 0 && pos.y <= game.count) {
                game_export(game_backward(game, pos.y), "game->");
            } else log("invalid input");
            continue;
        }
        if (!available(game.board, pos)) {
            log_i("time: %dms", get_time(tim));
            log_e("invalid position (%d, %d)!", pos.x, pos.y);
            // game_export(game, "game->");
            continue;
        } else {
            log_i("time: %dms, chose " BOLD UNDERLINE "(%c, %d)" NONE, get_time(tim), pos.y + 'A',
                  pos.x + 1);
        }
        if (game.cur_id == game.first_id) {
            int forbid = is_forbidden(game.board, pos, id, true);
            if (forbid) {
                log_e("forbidden position! (%s)", pattern4_typename[forbid]);
                continue;
                // game.winner = 3 - id;
                // return game;
            }
        }

        game_add_step(&game, pos);
        game_print(game);

        if (check_draw(game.board)) {
            game.winner = 0;
            return game;
        }
        if (check(game.board, pos)) {
            game.winner = id;
            return game;
        }
    }
    return game;
}

int results[5];

void signal_handler(int signum)
{
    log_s("");
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
struct {
    int p1, p2;
    int time_limit;
} preset_modes[PRESET_SIZE] = {
    {MANUAL, MCTS, GAME_TIME_LIMIT},
    {MCTS, MANUAL, GAME_TIME_LIMIT},
    {MANUAL, MANUAL, -1},
};

int main(void)
{
    signal(SIGINT, signal_handler);

    log("gomoku v%s", VERSION);

    init();

    log_i("available modes: ");
    for (int i = 0; i < PRESET_SIZE; i++) {
        if (preset_modes[i].time_limit > 0)
            log_i("#%d: %s vs %s\t%dms", i + 1, preset_players[preset_modes[i].p1].name,
                  preset_players[preset_modes[i].p2].name, preset_modes[i].time_limit);
        else
            log_i("#%d: %s vs %s", i + 1, preset_players[preset_modes[i].p1].name,
                  preset_players[preset_modes[i].p2].name);
    }
    log_i("#0: custom");

    int mode = -1;
    do {
        prompt(), scanf("%d", &mode);
    } while (mode < 0 || mode > PRESET_SIZE);

    int player1, player2, time_limit;
    if (!mode) {
        log_i("available players:");
        for (int i = 0; i < PLAYER_CNT; i++) log_i("#%d: %s", i, preset_players[i].name);
        log_i("input player1 player2:");
        do prompt(), scanf("%d%d", &player1, &player2);
        while (player1 < 0 || player1 >= PLAYER_CNT || player2 < 0 || player2 >= PLAYER_CNT);
        log_i("input time limit (-1 if no limit): ");
        prompt(), scanf("%d", &time_limit);
    } else {
        player1 = preset_modes[mode - 1].p1, player2 = preset_modes[mode - 1].p2;
        time_limit = preset_modes[mode - 1].time_limit;
    }

    int id = 1;
    while (1) {
        game_t game = start_game(preset_players[player1], preset_players[player2], id, time_limit);
        if (game.winner) {
            log_i("player%d wins", game.winner);
        } else {
            log_i("draw");
        }
        results[game.winner]++;
        if (game.winner == id) {
            results[3]++;
        } else if (game.winner == 3 - id) {
            results[4]++;
        }
        log_i("results: p1/p2/draw: %d/%d/%d, 1st/2nd: %d/%d", results[1], results[2], results[0],
              results[3], results[4]);
        id = 3 - id;
    }
    return 0;
}
