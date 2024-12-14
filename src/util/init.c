#include "pattern.h"
#include "players.h"
#include "zobrist.h"

#include <stdlib.h>
#include <time.h>

/// @brief initialize
void init()
{
    srand(time(NULL));
    zobrist_init();
    pattern_init();
    player_init();
    // log("initialized");
}