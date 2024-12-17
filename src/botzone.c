// author: Cauphenuny
// date: 2024/09/20

#include "board.h"
#include "game.h"
#include "init.h"
#include "players.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>

void read_pos(point_t* p)
{
    int x, y;
    scanf("%d%d", &x, &y);
    p->x = x, p->y = y;
}

void print_pos(point_t p)
{
    printf("%d %d\n", p.x, p.y);
}

int main()
{
    const double tim = record_time();
    log_lock();
    init();

    game_t game = new_game(990 - get_time(tim));

    int n;
    scanf("%d", &n);
    point_t p;
    for (int i = 0; i < 2 * n - 1; i++) {
        read_pos(&p);
        if (in_board(p)) {
            add_step(&game, p);
        }
    }
    const player_t player = preset_players[MCTS];
    p = player.move(game, player.assets);
    print_pos(p);
    log_flush();

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
