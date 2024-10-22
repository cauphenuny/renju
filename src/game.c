// author: Cauphenuny
// date: 2024/09/22

#include "game.h"

#include <stdio.h>
#include <string.h>

/// @brief create a new game, player{first_id} moves first, time limit {time_limit}ms
game_t game_new(int first_id, int time_limit)
{
    game_t game = {0};
    game.first_id = game.cur_id = first_id;
    if (time_limit > 0) game.time_limit = time_limit;
    else game.time_limit = 0x7f7f7f7f;
    return game;
}

/// @brief add a step to {game} at {pos}
void game_add_step(game_t* game, point_t pos)
{
    put(game->board, game->cur_id, pos);
    game->steps[game->count++] = pos;
    game->cur_id = 3 - game->cur_id;
}

/// @brief generate a game with the first {count} steps of {game}
/// @return generated game
game_t game_backward(game_t game, int count)
{
    game_t subgame = game_new(game.first_id, game.time_limit);
    for (int i = 0; i < count; i++) {
        game_add_step(&subgame, game.steps[i]);
    }
    return subgame;
}

/// @brief print the current board of {game}
void game_print(game_t game)
{
    if (game.count == 0) {
        print(game.board);
        return;
    }
    point_t pos = game.steps[game.count - 1];
    game.board[pos.x][pos.y] += 2;
    emph_print(game.board, pos);
    game.board[pos.x][pos.y] -= 2;
}

/// @brief create a game of given parameters
/// @param time_limit time limit
/// @param first_id first player
/// @param count count of existed pieces
/// @param moves positions of each piece
game_t game_import(int time_limit, int first_id, int count, point_t moves[])
{
    game_t game = game_new(first_id, time_limit);
    for (int i = 0; i < count; i++) {
        game_add_step(&game, moves[i]);
    }
    return game;
}

/// @brief export the state of {game} to {file}
/// @param file file to export, empty if uses stdout
void game_export(game_t game, const char* file)
{
    FILE* fp;
    if (strlen(file)) {
        fp = fopen(file, "w");
        if (fp == NULL) {
            perror("failed to open file");
            return;
        }
    } else {
        fp = stdout;
    }
    fprintf(fp, "game_import(%d,%d,%d,(point_t[]){", game.time_limit, game.first_id, game.count);
    for (int i = 0; i < game.count; i++) {
        fprintf(fp, "{%d,%d}", game.steps[i].x, game.steps[i].y);
        if (i != game.count - 1) fprintf(fp, ",");
    }
    fprintf(fp, "});\n");
    fclose(fp);
}