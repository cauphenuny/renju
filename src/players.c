// author: Cauphenuny
// date: 2024/07/26
#include "players.h"

#include "board.h"
#include "game.h"
#include "manual.h"
#include "mcts.h"
#include "mcts_nn.h"
#include "minimax.h"

#include <string.h>

mcts_parm_t mcts_preset = {
    .C = 1.414,
    .start_c = 3,
    .end_c = 0.5,
    .min_time = 200,
    .max_time = GAME_TIME_LIMIT,
    .min_count = 200,
    .wrap_rad = 2,
};

const char* player_name[PLAYER_CNT] = {
    "manual",
    "mcts",
    "mcts2",
    "mcts_nn",
    "minimax",
    "mix",
};

/// @brief generate next step by playerid and game info
point_t move(int player_type, void* player_assets, const game_t game)
{
    switch (player_type) {
    case MANUAL: {
        return manual();
    }
    case MCTS: {
        if (player_assets == NULL)
            return mcts(game, mcts_preset);
        else
            return mcts(game, *(mcts_parm_t*)player_assets);
    }
    case MCTS2: {
        if (player_assets == NULL) {
            mcts_parm_t parm = mcts_preset;
            return mcts(game, parm);
        } else {
            return mcts(game, *(mcts_parm_t*)player_assets);
        }
    }
    case MCTS_NN: {
        mcts_nn_parm_t parm;
        parm.network = player_assets;
        return mcts_nn(game, parm);
    }
    case MINIMAX: {
        return minimax(game);
    }
    case MIX: {
        if (game.count < 20)
            return minimax(game);
        else
            return mcts(game, mcts_preset);
    }
    }
    return (point_t){-1, -1};
}
