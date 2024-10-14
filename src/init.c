#include "board.h"
#include "util.h"
#include "zobrist.h"
void init()
{
    zobrist_init();
    // log("initialized zobrist");
    pattern_init();
    // log("initialized pattern");
}