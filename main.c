// author: Cauphenuny <https://cauphenuny.github.io/>
// date: 2024/07/27
#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <signal.h>
#include <stdbool.h>

#include "players.h"
#include "server.h"
#include "util.h"

int results[5];

void signal_handler(int signum) {
    log("received signal %d, terminate.", signum);
    int r1 = results[1], r2 = results[2];
    if (r1 + r2) {
        log("results: p1/p2/draw: %d/%d/%d (%.2lf%%), 1st/2nd: %d/%d (%.2lf%%)", r1, r2, results[0], (double)r1 / (r1 + r2) * 100, results[3], results[4], (double)results[3] / (r1 + r2) * 100);
    }
    exit(0);
}

int main(void) {
    signal(SIGINT, signal_handler);
    log("gomoku v%s", VERSION);
    int id = 1;
    int player1 = MCTS, player2 = MANUAL;
    while (1) {
        int res = game(player1, player2, id);
        results[res]++;
        if (res == id) {
            results[3]++;
        } else if (res == 3 - id) {
            results[4]++;
        }
        log("results: p1/p2/draw: %d/%d/%d, 1st/2nd: %d/%d", results[1], results[2], results[0], results[3], results[4]);
        id = 3 - id;
    }
    return 0;
}
