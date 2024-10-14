// author: Cauphenuny
// date: 2024/09/20

#include "board.h"
#include "game.h"
#include "init.h"
#include "players.h"
#include "util.h"

#include <stdio.h>

void read_pos(point_t* p)
{
    scanf("%hhd%hhd", &(p->x), &(p->y));
}

void print_pos(point_t p)
{
    printf("%hhd %hhd\n", p.x, p.y);
}

int main()
{
    init();

    game_t game = new_game(1, 990);

    int n;
    scanf("%d", &n);
    point_t p;
    read_pos(&p);
    if (inboard(p)) {
        game_add_step(&game, p);
    }
    player_t player = preset_players[MCTS];

    while (1) {
        p = player.move(game, player.assets);

        game_add_step(&game, p);

        print_pos(p);
        log_flush();

        printf("\n>>>BOTZONE_REQUEST_KEEP_RUNNING<<<\n");
        fflush(stdout);

        read_pos(&p);
        game_add_step(&game, p);
    }
}
