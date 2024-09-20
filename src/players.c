// author: Cauphenuny <https://cauphenuny.github.io/>
// date: 2024/07/26
#include "players.h"

#include "board.h"
#include "game.h"
#include "manual.h"
#include "mcts.h"

#include <string.h>

mcts_assets_t mcts1, mcts2;

void players_init()
{
    memset(&mcts1, 0, sizeof(mcts1));
    memset(&mcts2, 0, sizeof(mcts1));
    mcts_parm_t parm = {
        .C = 1.414,
        .M = 0,
        .MIN_TIME = 10,
        .MAX_TIME = 1000,
        .MIN_COUNT = 60,
        .WRAP_RAD = 2,
    };
    mcts1.mcts_parm = parm;
    // parm.WRAP_RAD = 1;
    mcts2.mcts_parm = parm;
    // assets_init(&mcts1);
    // assets_init(&mcts2);
}

point_t move(int player_type, const game_t game)
{
    switch (player_type) {
    case MANUAL: {
        return manual(game);
    }
    case MCTS: {
        return mcts(game, &mcts1);
    }
    case MCTS2: {
        return mcts(game, &mcts2);
    }
    }
    return (point_t){-1, -1};
}
