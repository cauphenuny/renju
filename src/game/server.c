// author: Cauphenuny
// date: 2024/10/19

#include "board.h"
#include "eval.h"
#include "game.h"
#include "pattern.h"
#include "players.h"
#include "util.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

/// @brief start a game with player {p1} and {p2}, {p{first_id}} move first
/// @param time_limit game time limit
game_result_t start_game(player_t p1, player_t p2, int first_id, int time_limit, network_t* network)
{
    player_t players[] = {{}, p1, p2};
    const char* colors[] = {"", L_GREEN, L_RED};
    log("start game: %s vs %s, first player: %d", p1.name, p2.name, first_id);
    game_t game = new_game(time_limit);
    game_result_t result = {0};
    print_game(game);
#define WIN(winner_id)                                 \
    do {                                               \
        result.game = game, result.winner = winner_id; \
        return result;                                 \
    } while (0)
    int id = first_id;
    while (1) {
        // log_disable();
        log_i("------ step %s#%d" RESET ", player%d's turn ------", colors[game.cur_id],
              game.count + 1, id);
        if (!have_space(game.board, game.cur_id)) {
            log("no more space for player%d", id);
            WIN(3 - id);
        }
        bind_output_prob(result.prob[game.count]);
        const int tim = record_time();
        const point_t pos = players[id].move(game, players[id].assets);

        switch (pos.x) {
            case GAMECTRL_WITHDRAW:
                if (pos.y > 0 && pos.y <= game.count) {
                    game = backward(game, game.count - pos.y);
                    print_game(game);
                } else
                    log_e("invalid argument!");
                continue;
            case GAMECTRL_EXPORT:
                if (pos.y > 0 && pos.y <= game.count) {
                    serialize_game(backward(game, pos.y), "");
                } else
                    log_e("invalid argument!");
                continue;
            case GAMECTRL_GIVEUP: log("player %d gave up.", id); WIN(3 - id);
            case GAMECTRL_SWITCH_PLAYER:
                if (pos.y >= 0 && pos.y < PLAYER_CNT) {
                    players[3 - id] = preset_players[pos.y];
                    log("changed opponent to %s", players[3 - id].name);
                } else {
                    log("invalid argument!");
                }
                continue;
            case GAMECTRL_EVALUATE:
                if (!network) {
                    log_e("no network available!");
                    continue;
                } else {
                    prediction_t prediction =
                        predict(network, game.board, game.steps[game.count - 1], game.cur_id);
                    if (pos.y) {
                        print_prediction(prediction);
                    } else {
                        log("eval: %.3lf", prediction.eval);
                    }
                    continue;
                }
            default: break;
        }
        if (!available(game.board, pos)) {
            log_e("invalid position!");
            continue;
        }
        log_i("time: %dms, chose " BOLD UNDERLINE "(%c, %d)" RESET, get_time(tim), pos.y + 'A',
              pos.x + 1);
#ifndef NO_FORBID
        if (game.cur_id == 1) {
            const int forbid = is_forbidden(game.board, pos, game.cur_id, true);
            // int forbid = false;
            if (forbid) {
                log_e("forbidden position! (%s)", pattern4_typename[forbid]);
                prompt_pause();
                continue;
                // WIN(3 - id);
            }
        }
#endif

        add_step(&game, pos);
        print_game(game);
        // serialize_game(game, "");
        // if (network != NULL) {
        //     prediction_t pred = predict(network, game.board, game.first_id, 3 - game.cur_id);
        //     log_i("evaluate: %f",
        //           pred.eval *
        //               (game.first_id == 1 ? 1 : -1));  // for 1.00 -> 'o' wins, -1.00 -> 'x' wins
        //     // print_prob(game.board, pred.prob);
        // }

        if (is_draw(game.board)) WIN(0);
        if (check(game.board, pos)) WIN(id);
        id = 3 - id;
    }
}
