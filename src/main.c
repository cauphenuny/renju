#include "dataset.h"
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

#define GAME_STORAGE_SIZE 65536
#define DATASET_SIZE      65536

int statistics[3][3];
// [0][0]: draw count [0][1/2]: normal / reverse count
// [1][0/1/2]: p1 win count (all / 1st / 2nd)
// [2][0/1/2]: p2 win count (all / 1st / 2nd)

static void print_statistics(void) {
    const int sum = statistics[0][1] + statistics[0][2];
    if (sum) {
        const double total_base = 100.0 / sum;
        log_i("statistics:");
        log_i("winner: p1/p2/draw: %d/%d/%d (%.2lf%% - %.2lf%%)", statistics[1][0],
              statistics[2][0], statistics[0][0], statistics[1][0] * total_base,
              statistics[2][0] * total_base);
        const double normal_base = 100.0 / statistics[0][1],
                     reverse_base = 100.0 / statistics[0][2];
        log_i("player1: win when play as 1st: %d/%d (%.2lf%%), 2nd: %d/%d (%.2lf%%)",
              statistics[1][1], statistics[0][1], statistics[1][1] * normal_base, statistics[1][2],
              statistics[0][2], statistics[1][2] * reverse_base);
        log_i("player2: win when play as 1st: %d/%d (%.2lf%%), 2nd: %d/%d (%.2lf%%)",
              statistics[2][1], statistics[0][2], statistics[2][1] * reverse_base, statistics[2][2],
              statistics[0][1], statistics[2][2] * normal_base);
    }
}

static game_result_t results[GAME_STORAGE_SIZE];
static int tot;
static char* sample_file;
static char* model_file;
static network_t network;
static dataset_t dataset;

static void save_data() {
    log("save game data? [y/n]");
    int c;
    do c = prompt_pause();
    while (c != 'y' && c != 'n');
    if (c == 'y') {
        char name[256];
        log("transform? [y/n]");
        do c = prompt_pause();
        while (c != 'y' && c != 'n');
        if (c == 'y')
            add_games(&dataset, results, tot);
        else
            add_testgames(&dataset, results, tot);
        do {
            if (!sample_file) {
                log("input file name: ");
                prompt_scanf("%s", name);
            } else {
                log("input file name (empty for %s): ", sample_file);
                int first = prompt_pause();
                if (first == '\n' || first == EOF) {
                    strcpy(name, sample_file);
                } else {
                    name[0] = first;
                    chkscanf("%s", name + 1);
                }
            }
        } while (save_dataset(&dataset, name));
    }
}

static void signal_handler(int signum) {
    switch (signum) {
        case SIGINT:
            log_s("\n");
            log("received signal SIGINT, terminate.");
            print_statistics();
            if (tot) {
                save_data();
            }
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
    const char* name;
} preset_modes[PRESET_SIZE] = {
    {MANUAL, MINIMAX_ULT, GAME_TIME_LIMIT, "player first"},
    {MINIMAX_ULT, MANUAL, GAME_TIME_LIMIT, "AI first"},
    {MANUAL, MANUAL, -1, "pvp, no AI"},
};

int main(int argc, char* argv[]) {
    signal(SIGINT, signal_handler);

    log("renju v%s", VERSION);
    init();

#ifdef DEFAULT_MODEL
    char fullname[256];
    snprintf(fullname, 256, "%s.v%d.%dch.mod", DEFAULT_MODEL, NETWORK_VERSION, MAX_CHANNEL);
    if (file_exists(fullname)) {
        model_file = fullname;
        load_network(&network, model_file);
        bind_network(&network, false);
    }
#endif

    if (argc > 1) {
        for (int i = 1; i < argc; i++) {
            char* p = argv[i] + strlen(argv[i]);
            while (*p != '.' && p > argv[i]) p--;
            if (strcmp(p, ".dat") == 0 && !sample_file && file_exists(argv[i])) {
                sample_file = argv[i], load_dataset(&dataset, sample_file);
            } else if (strcmp(p, ".mod") == 0 && !model_file) {
                if (!load_network(&network, argv[i])) {
                    bind_network(&network, false);
                    model_file = argv[i];
                } else {
                    log_e("invalid model: %s", argv[i]);
                }
            } else {
                log_e("invalid argument: %s", argv[i]);
            }
        }
    }

    if (dataset.capacity == 0) {
        dataset = new_dataset(DATASET_SIZE);
    }

    if (model_file == NULL) {
        log_e("network not found. usage: %s [.mod file]", argv[0]);
    }

    int player1, player2, time_limit;
#if (DEBUG_LEVEL > 1) || defined(TEST)
    player1 = MINIMAX_ADV, player2 = MINIMAX_ADV, time_limit = 15000;
#else
    log_i("available modes: ");
    for (int i = 0; i < PRESET_SIZE; i++) {
        log_i("%d: %s", i + 1, preset_modes[i].name);
    }
    log_i("0: custom");
    log_i("input mode:");

    int mode = -1;
    do {
        prompt_scanf("%d", &mode);
    } while (mode < 0 || mode > PRESET_SIZE);
    if (!mode) {
        log_i("player presets:");
        for (int i = 0; i < PLAYER_CNT; i++) log_i("%d: %s", i, preset_players[i].name);
        log_i("input player1 player2 (%%d %%d):");
        do {
            prompt();
            chkscanf("%d%d", &player1, &player2)
        } while (player1 < 0 || player1 >= PLAYER_CNT || player2 < 0 || player2 >= PLAYER_CNT);
        log_i("input time limit (unit: ms) (-1 if no limit): ");
        prompt_scanf("%d", &time_limit);
    } else {
        player1 = preset_modes[mode - 1].p1, player2 = preset_modes[mode - 1].p2;
        time_limit = preset_modes[mode - 1].time_limit;
    }
#endif

    int id = 1;
    while (1) {
        const game_result_t result = start_game(preset_players[player1], preset_players[player2],
                                                id, time_limit, model_file ? &network : NULL);
        if (tot < GAME_STORAGE_SIZE) results[tot++] = result;
        const int winner = result.winner;
        statistics[0][id]++;
        if (winner) {
            log_i("player%d (%s) wins", winner, winner == id ? "first" : "second");
            statistics[winner][0]++;
            statistics[winner][(winner != id) + 1]++;
        } else {
            log_i("draw");
            statistics[0][0]++;
        }
        print_statistics();
        id = 3 - id;
        // save_data();
    }
    return 0;
}
