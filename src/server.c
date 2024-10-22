// author: Cauphenuny
// date: 2024/10/19

#include "players.h"
#include "util.h"

/// @brief start a game with player {p1} and {p2}, {p{first_id}} move first
/// @param time_limit game time limit
int start_game(player_t p1, player_t p2, int first_id, int time_limit)
{
    player_t players[] = {{}, p1, p2};
    const char* colors[] = {"", L_GREEN, L_RED};
    log("start game: %s vs %s, first player: %d", p1.name, p2.name, first_id);
    game_t game = game_new(first_id, time_limit);
    point_t pos;
    game_print(game);
    while (1) {
        int id = game.cur_id;
        int tim;
        log_i("------ step %s#%d" RESET ", player%d's turn ------", colors[id], game.count + 1, id);
        tim = record_time();
        game_export(game, "gomoku.log");
        pos = players[id].move(game, players[id].assets);

        switch (pos.x) {
            case GAMECTRL_WITHDRAW:
                if (pos.y > 0 && pos.y <= game.count) {
                    game = game_backward(game, game.count - pos.y);
                    game_print(game);
                } else
                    log("invalid input");
                continue;
            case GAMECTRL_EXPORT:
                if (pos.y > 0 && pos.y <= game.count) {
                    game_export(game_backward(game, pos.y), "");
                } else
                    log("invalid input");
                continue;
            case GAMECTRL_GIVEUP: log("player %d gave up.", id); return 3 - id;
            default: break;
        }
        if (!available(game.board, pos)) {
            log_i("time: %dms", get_time(tim));
            log_e("invalid position!"), prompt_pause();
            continue;
        } else {
            log_i("time: %dms, chose " BOLD UNDERLINE "(%c, %d)" RESET, get_time(tim), pos.y + 'A',
                  pos.x + 1);
        }
        if (game.cur_id == game.first_id) {
            int forbid = is_forbidden(game.board, pos, id, true);
            if (forbid) {
                log_e("forbidden position! (%s)", pattern4_typename[forbid]);
                // prompt_pause();
                // continue;
                return 3 - id;
            }
        }

        game_add_step(&game, pos);
        // game_export(game);
        game_print(game);

        if (is_draw(game.board)) {
            return 0;
        }
        if (check(game.board, pos)) {
            return id;
        }
        if (!have_space(game.board, game.cur_id)) {
            log("no more space for player%d", game.cur_id);
            return 3 - game.cur_id;
        }
    }
}
