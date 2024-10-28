// author: Cauphenuny
// date: 2024/07/27
#include "game.h"
#include "init.h"
#include "players.h"
#include "server.h"
#include "util.h"

#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int statistics[3][3];
// [0][0]: draw count [0][1/2]: normal / reverse count
// [1][0/1/2]: p1 win count (all / 1st / 2nd)
// [2][0/1/2]: p2 win count (all / 1st / 2nd)

static void print_statistics(void)
{
    const int sum = statistics[0][1] + statistics[0][2];
    if (sum) {
        const double total_base = 100.0 / sum;
        log_i("statistics:");
        log_i("winner: p1/p2/draw: %d/%d/%d (%.2lf%% - %.2lf%%)", statistics[1][0],
              statistics[2][0], statistics[0][0], statistics[1][0] * total_base,
              statistics[2][0] * total_base);
        const double normal_base = 100.0 / statistics[0][1], reverse_base = 100.0 / statistics[0][2];
        log_i("player1: win when play as 1st: %d/%d (%.2lf%%), 2nd: %d/%d (%.2lf%%)",
              statistics[1][1], statistics[0][1], statistics[1][1] * normal_base, statistics[1][2],
              statistics[0][2], statistics[1][2] * reverse_base);
        log_i("player2: win when play as 1st: %d/%d (%.2lf%%), 2nd: %d/%d (%.2lf%%)",
              statistics[2][1], statistics[0][2], statistics[2][1] * reverse_base, statistics[2][2],
              statistics[0][1], statistics[2][2] * normal_base);
    }
}

static void signal_handler(int signum)
{
    switch (signum) {
        case SIGINT:
            log_s("\n");
            log("received signal SIGINT, terminate.");
            print_statistics();
            exit(0);
        default: log_e("unexpected signal %d", signum);
    }
    exit(signum);
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
    // signal(SIGSEGV, signal_handler);

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

    int player1, player2, time_limit;
#ifndef DEBUG
    int mode = -1;
    do {
        prompt(), scanf("%d", &mode);
    } while (mode < 0 || mode > PRESET_SIZE);
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
#else
    player1 = MCTS, player2 = MCTS, time_limit = -1;
#endif

    int id = 1;
    while (1) {
        const int winner = start_game(preset_players[player1], preset_players[player2], id, time_limit);
        statistics[0][id]++;
        if (winner) {
            log_i("player%d wins", winner);
            statistics[winner][0]++;
            statistics[winner][(winner != id) + 1]++;
        } else {
            log_i("draw");
            statistics[0][0]++;
        }
        print_statistics();
        id = 3 - id;
    }
    return 0;
}
