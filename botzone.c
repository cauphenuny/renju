// author: Cauphenuny <https://cauphenuny.github.io/>
// date: 2024/09/20

#include "board.h"
#include "game.h"
#include "players.h"
#include "util.h"
#include "zobrist.h"

#include <stdio.h>

game_t game;

void read_pos(point_t* p)
{
    scanf("%hhd%hhd", &(p->x), &(p->y));
}

void print_pos(point_t p)
{
    printf("%hhd %hhd\n", p.x, p.y);
}

void read_multy_pos(int n)
{
    int id = 1;
    for (int i = 0; i < n; i++) {
        point_t p;
        read_pos(&p);
        if (inboard(p)) {
            put(game.board, id, p), refresh(game.board);
            game.steps[game.step_cnt++] = p;
        }
        if (!i) {
            if (inboard(p))
                game.first_id = id;
            else
                game.first_id = 3 - id;
        }
        id = 3 - id;
    }
}

int main()
{
    zobrist_init();
    players_init();

    int n;
    scanf("%d", &n);

    read_multy_pos(n * 2 - 1);

    point_t p;
    // point_t p = move(MCTS, game);

    // print_pos(p);

    while (1) {
        game.current_id = 2;
        p = move(MINIMAX, game);
        put(game.board, 2, p), refresh(game.board);
        game.steps[game.step_cnt++] = p;

        print_pos(p);
        log_flush();

        printf("\n>>>BOTZONE_REQUEST_KEEP_RUNNING<<<\n");
        fflush(stdout);

        read_pos(&p);
        put(game.board, 1, p), refresh(game.board);
        game.steps[game.step_cnt++] = p;
    }
}