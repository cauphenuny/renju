// author: Cauphenuny
// date: 2024/10/19

#include "game.h"
#include "neuro.h"
#include "players.h"
#include "util.h"
#include "pattern.h"

#include <string.h>

/// @brief start a game with player {p1} and {p2}, {p{first_id}} move first
/// @param time_limit game time limit
game_result_t start_game(player_t p1, player_t p2, int first_id, int time_limit,
                         checker_network_t* checker)
{
    const player_t players[] = {{}, p1, p2};
    const char* colors[] = {"", L_GREEN, L_RED};
    log("start game: %s vs %s, first player: %d", p1.name, p2.name, first_id);
    game_t game = game_new(first_id, time_limit);
    game_result_t result = {0};
    game_print(game);
#define WIN(winner_id)                                 \
    do {                                               \
        result.game = game, result.winner = winner_id; \
        return result;                                 \
    } while (0)
    while (1) {
        const int id = game.cur_id;
        log_i("------ step %s#%d" RESET ", player%d's turn ------", colors[id], game.count + 1, id);
        if (!have_space(game.board, id)) {
            log("no more space for player%d", id);
            WIN(3 - id);
        }
        mcts_params_default.prob = result.prob[game.count];
        const int tim = record_time();
        const point_t pos = players[id].move(game, players[id].assets);

        switch (pos.x) {
            case GAMECTRL_WITHDRAW:
                if (pos.y > 0 && pos.y <= game.count) {
                    game = game_backward(game, game.count - pos.y);
                    game_print(game);
                } else
                    log_e("invalid argument!");
                continue;
            case GAMECTRL_EXPORT:
                if (pos.y > 0 && pos.y <= game.count) {
                    game_serialize(game_backward(game, pos.y), "");
                } else
                    log_e("invalid argument!");
                continue;
            case GAMECTRL_GIVEUP: log("player %d gave up.", id); WIN(3 - id);
            default: break;
        }
        if (!available(game.board, pos)) {
            log_e("invalid position!");
            continue;
        }
        log_i("time: %dms, chose " BOLD UNDERLINE "(%c, %d)" RESET, get_time(tim), pos.y + 'A',
              pos.x + 1);
        if (game.cur_id == game.first_id) {
            const int forbid = is_forbidden(game.board, pos, id, true);
            // int forbid = false;
            if (forbid) {
                log_e("forbidden position! (%s)", pattern4_typename[forbid]);
                prompt_pause();
                // continue;
                WIN(3 - id);
            }
        }

        game_add_step(&game, pos);
#if DEBUG_LEVEL > 0
        game_serialize(game, "");
#endif
        game_print(game);
        if (checker != NULL) {
            int ret = checker_forward(checker, game.board);
            if (ret)
                log_w("checker returned: %d", ret);
            else
                log("checker returned: %d", ret);
        }

        if (is_draw(game.board)) WIN(0);
        if (check(game.board, pos)) WIN(id);
    }
}
