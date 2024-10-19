#include "board.h"
#include "players.h"
#include "zobrist.h"

/// @brief initialize
void init()
{
    zobrist_init();
    pattern_init();
    player_init();
    // log("initialized");
}