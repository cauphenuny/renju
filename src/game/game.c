#include "game.h"

#include <stdio.h>
#include <string.h>

game_t new_game(int time_limit) {
    game_t game = {0};
    game.cur_id = 1;
    if (time_limit > 0)
        game.time_limit = time_limit;
    else
        game.time_limit = 0x7f7f7f7f;
    return game;
}

void add_step(game_t* game, point_t pos) {
    put(game->board, game->cur_id, pos);
    game->steps[game->count++] = pos;
    game->cur_id = 3 - game->cur_id;
}

/// @brief generate a game with the first {count} steps of {game}
game_t backward(game_t game, int count) {
    game_t subgame = new_game(game.time_limit);
    for (int i = 0; i < count; i++) {
        add_step(&subgame, game.steps[i]);
    }
    return subgame;
}

/// @brief print the current board of {game}
void print_game(game_t game) {
    if (game.count == 0) {
        print(game.board);
        return;
    }
    const point_t pos = game.steps[game.count - 1];
    game.board[pos.x][pos.y] += 2;
    print_emph(game.board, pos);
    game.board[pos.x][pos.y] -= 2;
}

game_t restore_game(int time_limit, int count, point_t moves[]) {
    game_t game = new_game(time_limit);
    for (int i = 0; i < count; i++) {
        add_step(&game, moves[i]);
    }
    return game;
}

/// @brief export the state of {game} to {file}
/// @param file file to export, empty if uses stdout
void serialize_game(game_t game, const char* file) {
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
    fprintf(fp, "restore_game(%d,%d,(point_t[]){", game.time_limit, game.count);
    for (int i = 0; i < game.count; i++) {
        fprintf(fp, "{%d,%d}", game.steps[i].x, game.steps[i].y);
        if (i != game.count - 1) fprintf(fp, ",");
    }
    fprintf(fp, "});\n");
    if (fp != stdout) fclose(fp);
}
