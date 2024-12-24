#include "board.h"
#include "game.h"
#include "init.h"
#include "players.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>

#define BOTZONE_TIME_LIMIT 2000
#define BOTZONE_PLAYER     MINIMAX_ULT

void read_pos(point_t* p) {
    int x, y;
    scanf("%d%d", &x, &y);
    p->x = x, p->y = y;
}

void print_pos(point_t p) { printf("%d %d\n", p.x, p.y); }

int main() {
    const double tim = record_time();
    log_lock();
    init();
    game_t game = new_game(BOTZONE_TIME_LIMIT - (int)get_time(tim));

    int n;
    scanf("%d", &n);
    point_t p;
    int id = 1, is_first = 1;
    for (int i = 0; i < 2 * n - 1; i++) {
        read_pos(&p);
        if (in_board(p)) {
            if (is_first && is_forbidden(game.board, p, 1, -1)) {
                printf("-1 0\n");
                printf("forbidden\n");
                return 0;
            }
            add_step(&game, p);
            if (check(game.board, p)) {
                printf("-1 %d\n", id);
                printf("%s win\n", is_first == 1 ? "black" : "white");
                return 0;
            }
            is_first = !is_first;
        }
        id = 3 - id;
    }

    p = move(game, preset_players[BOTZONE_PLAYER]);
    print_pos(p);
    log_flush(true);

    // log_unlock();
    // print(game.board);

    // while (1) {
    //     p = player.move(game, player.assets);

    //     add_step(&game, p);

    //     print_pos(p);
    //     log_flush();

    //     printf("\n\n>>>BOTZONE_REQUEST_KEEP_RUNNING<<<\n");
    //     fflush(stdout);

    //     read_pos(&p);
    //     add_step(&game, p);
    // }
}
