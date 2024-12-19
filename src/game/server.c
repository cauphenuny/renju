#include "board.h"
#include "eval.h"
#include "game.h"
#include "pattern.h"
#include "players.h"
#include "threat.h"
#include "util.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

int vct_depth_sum, vct_depth_max, game_cnt;

/// @brief start a game with player {p1} and {p2}, {first_player} moves first
/// @param time_limit time limit of every step
game_result_t start_game(player_t p1, player_t p2, int first_player, int time_limit,
                         network_t* network) {
    game_cnt++;
    player_t players[] = {{}, p1, p2};
    const char* colors[] = {"", L_GREEN, L_RED};
    set_color(first_player == 1);
    game_result_t result = {0};
    int claim_winner = 0;
    int player = first_player;
#define WIN(winner_id)                                                                            \
    do {                                                                                          \
        if (claim_winner && players[claim_winner].attribute.enable_vct &&                         \
            winner_id != claim_winner) {                                                          \
            log_w("claim incorrect!");                                                            \
            prompt_pause();                                                                       \
        }                                                                                         \
        log("VCT sequence avg: %.2lf, max: %d", (double)vct_depth_sum / game_cnt, vct_depth_max); \
        result.game = game, result.winner = winner_id;                                            \
        return result;                                                                            \
    } while (0)
    log("start game: [%s] vs [%s], %dms, first player: %d", p1.name, p2.name, time_limit, first_player);
    game_t game = new_game(time_limit);
    print_game(game);
    while (1) {
        // log_disable();
        log_i("------ step %s#%d" RESET ", player%d[%s]'s turn ------", colors[player],
              game.count + 1, player, players[player].name);
        if (!have_space(game.board, game.cur_id)) {
            log("no more space for player%d", player);
            WIN(3 - player);
        }
        bind_output_prob(result.prob[game.count]);
        const double tim = record_time();
        const point_t pos = players[player].move(game, players[player].assets);
        const double duration = get_time(tim);

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
            case GAMECTRL_GIVEUP: log("player%d gave up.", player); WIN(3 - player);
            case GAMECTRL_SWITCH_PLAYER:
                if (pos.y >= 0 && pos.y < PLAYER_CNT) {
                    players[3 - player] = preset_players[pos.y];
                    log("changed opponent to %s", players[3 - player].name);
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
            prompt_pause();
            continue;
        }
        log_i("time: %.2lfms, chose " BOLD UNDERLINE "%c%d" RESET, duration, READABLE_POS(pos));
        if (duration > game.time_limit * 1.5 && !players[player].attribute.no_time_limit) {
            log_e("timeout.");
            prompt_pause();
        }
#ifndef NO_FORBID
        if (game.cur_id == 1) {
            const int forbid = is_forbidden(game.board, pos, game.cur_id, -1);
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

        serialize_game(game, "");
        log("eval: %lld", eval(game.board));

        if (is_draw(game.board)) WIN(0);
        if (check(game.board, pos)) WIN(player);
        player = 3 - player;

        if (!claim_winner) {
            vector_t vct_sequence = vct(false, game.board, game.cur_id, 50);
            if (vct_sequence.size) {
                print_points(vct_sequence, PROMPT_NOTE, "->");
                vct_depth_sum += vct_sequence.size;
                vct_depth_max = max(vct_depth_max, (int)vct_sequence.size);
                log_w("found VCT sequence, claim p%d will win", player);
                claim_winner = player;
                if (vct_sequence.size < 3) prompt_pause();
            }
            vector_free(vct_sequence);
        }
    }
}
