// author: Cauphenuny
// date: 2024/09/22

#include "game.h"

#include "util.h"

#include <stdio.h>

game_t new_game(int first_id)
{
    game_t game = {0};
    game.first_player = game.cur_player = first_id;
    return game;
}

void game_add_step(game_t* game, point_t pos)
{
    put(game->board, game->cur_player, pos);
    game->steps[game->count++] = pos;
    game->cur_player = 3 - game->cur_player;
}

game_t game_backward(game_t game, int after_step)
{
    game_t subgame = new_game(game.first_player);
    for (int i = 0; i < after_step; i++) {
        game_add_step(&subgame, game.steps[i]);
    }
    return subgame;
}

void game_print(game_t game)
{
    if (game.count == 0) {
        print(game.board);
        return;
    }
    point_t pos = game.steps[game.count - 1];
    game.board[pos.x][pos.y] += 2;
    print(game.board);
    game.board[pos.x][pos.y] -= 2;
}

void game_export(game_t game, const char* name)
{
    int id = game.first_player;
    log("start export");
    printf("%sfirst_id=%d;", name, id);
    printf("%sstep_cnt=%d", name, game.count);
    for (int i = 0, x, y; i < game.count; i++) {
        x = game.steps[i].x;
        y = game.steps[i].y;
        printf("%sboard[%d][%d]=%d;", name, x, y, id);
        printf("%ssteps[%d]=(point_t){%d, %d};", name, i, x, y);
        id = 3 - id;
    }
    printf("%scurrent_id=%d;", name, id);
}