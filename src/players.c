// author: Cauphenuny <https://cauphenuny.github.io/>
// date: 2024/07/26
#include "players.h"

#include "board.h"
#include "game.h"
#include "manual.h"
#include "mcts.h"
#include "minimax.h"

#include <string.h>

mcts_parm_t mcts1, mcts2;

/// @brief initialize players
void players_init()
{
    memset(&mcts1, 0, sizeof(mcts1));
    memset(&mcts2, 0, sizeof(mcts1));
    mcts_parm_t parm = {
        .C = 1.414,
        .MIN_TIME = 500,
        .MAX_TIME = GAME_TIME_LIMIT,
        .MIN_COUNT = 1000,
        .WRAP_RAD = 2,
    };
    mcts1 = parm;
    parm.WRAP_RAD = 3;
    mcts2 = parm;
    // assets_init(&mcts1);
    // assets_init(&mcts2);
}

/// @brief generate next step by playerid and game info
point_t move(int player_type, const game_t game)
{
    switch (player_type) {
    case MANUAL: {
        return manual();
    }
    case MCTS: {
        return mcts(game, mcts1);
    }
    case MCTS2: {
        return mcts(game, mcts2);
    }
    case MINIMAX: {
        return minimax(game);
    }
    case MIX: {
        if (game.step_cnt < 15) return minimax(game);
        else return mcts(game, mcts1);
    }
    }
    return (point_t){-1, -1};
}
