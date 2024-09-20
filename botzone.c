// author: Cauphenuny <https://cauphenuny.github.io/>
// date: 2024/07/27

#include "board.h"
#include "game.h"
#include "players.h"
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

int main()
{
    zobrist_init();
    players_init();

    int n;
    scanf("%d", &n);
    point_t p;
    read_pos(&p);
    if (inboard(p)) {
        put(game.board, 1, p);
        game.steps[game.step_cnt++] = p;
    }
    refresh(game.board);

    game.current_id = 2;
    point_t pos = move(MCTS, game);
    print_pos(pos);
    put(game.board, 2, pos);
    game.steps[game.step_cnt++] = p;
    printf("\n>>>BOTZONE_REQUEST_KEEP_RUNNING<<<\n");

    // print(board);
    fflush(stdout);
    while (1) {
        read_pos(&p);
        put(game.board, 1, p);
        game.steps[game.step_cnt++] = p;
        refresh(game.board);

        game.current_id = 2;
        pos = move(MCTS, game);
        put(game.board, 2, pos);
        game.steps[game.step_cnt++] = pos;
        print_pos(pos);
        printf("\n>>>BOTZONE_REQUEST_KEEP_RUNNING<<<\n");
        // print(board);
        fflush(stdout);
    }
}