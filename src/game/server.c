/// @file server.c
/// @brief implementation of server

#include "board.h"
#include "game.h"
#include "pattern.h"
#include "players.h"
#include "util.h"
#include "vct.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#if DEBUG_LEVEL > 0 && \
    (defined(__unix__) || defined(__unix) || (defined(__APPLE__) && defined(__MACH__)))
#define BACKTRACE 1
#else
#define BACKTRACE 0
#endif

#if BACKTRACE
#include <execinfo.h>
#include <setjmp.h>
#include <signal.h>
#include <unistd.h>

static jmp_buf env;
static void (*original_handler)(int);

void signal_handler(int sig) {
    if (sig == SIGALRM) {
        void* array[20];
        size_t size;

        size = backtrace(array, 20);
        fprintf(stderr, "\ntimeout, stack trace:\n");
        backtrace_symbols_fd(array, size, STDERR_FILENO);

        while (1);
        longjmp(env, 1);
    }
}

void start_timer(int time_limit) {
    original_handler = signal(SIGALRM, signal_handler);
    alarm(time_limit);
}

void stop_timer() {
    alarm(0);
    signal(SIGALRM, original_handler);
}
#endif

int vct_depth_sum, vct_depth_cur, vct_depth_max, game_cnt;

/// @brief start a game with player {p1} and {p2}, {first_player} moves first
/// @param time_limit time limit of every step
game_result_t start_game(player_t p1, player_t p2, int first_player, int time_limit, game_t* start,
                         network_t* network) {

    const char* colors[] = {"", L_BLUE, L_RED};
    set_color(first_player == 1);

    game_cnt++, vct_depth_cur = 0;
    player_t players[] = {{}, p1, p2};
    game_result_t result = {0};
    int claim_winner = 0;
    int player = first_player;

#define WIN(winner_id)                                                    \
    do {                                                                  \
        if (claim_winner && players[claim_winner].attribute.enable_vct && \
            winner_id != claim_winner) {                                  \
            log_w("claim incorrect!");                                    \
            prompt_pause();                                               \
        }                                                                 \
        log_l("VCT sequence cur: %d, avg: %.2lf, max: %d", vct_depth_cur, \
              (double)vct_depth_sum / game_cnt, vct_depth_max);           \
        result.game = game, result.winner = winner_id;                    \
        return result;                                                    \
    } while (0)

    game_t game;
    if (!start)
        game = new_game(time_limit);
    else
        game = *start;
    log_l("[%s] vs [%s], time_limit: %dms", p1.name, p2.name, time_limit);
    print_game(game);
    while (1) {
        log_i("------ step %s#%d" RESET ", player%d[%s]'s turn ------", colors[player],
              game.count + 1, player, players[player].name);
        if (!have_space(game.board, game.cur_id)) {
            log_l("no more space for player%d", player);
            WIN(3 - player);
        }
        bind_output_prob(result.prob[game.count]);

#if BACKTRACE
        if (setjmp(env) == 0) {
            if (!players[player].attribute.allow_timeout) {
                // set timeout limit to 1000/750 = 1.33 times of origininal limit
                start_timer(time_limit / 750);
            }
#endif
            const double tim = record_time();
            const point_t pos = players[player].move(game, players[player].assets);
            const double duration = get_time(tim);
#if BACKTRACE
            if (!players[player].attribute.allow_timeout) {
                stop_timer();
            }
#endif

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
                case GAMECTRL_GIVEUP: log_l("player%d gave up.", player); WIN(3 - player);
                case GAMECTRL_SWITCH_PLAYER:
                    if (pos.y >= 0 && pos.y < PLAYER_CNT) {
                        players[3 - player] = preset_players[pos.y];
                        log_l("changed opponent to %s", players[3 - player].name);
                    } else {
                        log_l("invalid argument!");
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
                            print_prob(game.board, prediction.prob);
                            log_l("eval: %.3lf", prediction.eval);
                        } else {
                            log_l("eval: %.3lf", prediction.eval);
                        }
                        continue;
                    }
                default: break;
            }
            if (!available(game.board, pos)) {
                log_e("invalid position!");
                continue;
            }
            log_i("time: %.2lfms, chose " BOLD UNDERLINE "%c%d" RESET, duration, READABLE_POS(pos));
            if (duration > game.time_limit * 1.5 && !players[player].attribute.allow_timeout) {
                log_e("timeout.");
                prompt_pause();
            }
#ifndef NO_FORBID
            if (game.cur_id == 1) {
                const int forbid = is_forbidden(game.board, pos, game.cur_id, -1);
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
            // log_l("eval: %d", eval(game.board));

            if (is_draw(game.board)) WIN(0);
            if (check(game.board, pos)) WIN(player);
            player = 3 - player;

            if (!claim_winner) {
                for (int vct_depth = 0; vct_depth <= 2 && !claim_winner; vct_depth++) {
                    double vct_tim = record_time();
                    vector_t vct_sequence =
                        complex_vct(false, game.board, game.cur_id, 1000, vct_depth);
                    vct_tim = get_time(vct_tim);
                    if (vct_sequence.size) {
                        print_points(vct_sequence, PROMPT_NOTE, " -> ");
                        vct_depth_cur = vct_sequence.size;
                        vct_depth_sum += vct_sequence.size;
                        vct_depth_max = max(vct_depth_max, (int)vct_sequence.size);
                        log_w("found VCT sequence (depth: %d, %.2lfms), claim p%d will win",
                              vct_depth, vct_tim, player);
                        claim_winner = player;
                    }
                    vector_free(vct_sequence);
                }
            }
#if BACKTRACE
        } else {
            fprintf(stderr, "player move function execution exceeded time limit.\n");
            exit(EXIT_FAILURE);
        }
#endif
    }
}
