#include "assert.h"
#include "board.h"
#include "init.h"
#include "util.h"

static int test_forbid(void)
{
    int n = 6;
    struct {
        board_t board;
        point_t pos;
        int id;
    } tests[10] = {
        {
            {
                {0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 1, 0, 0, 0},
                {0, 0, 0, 1, 0, 0, 0},
                {0, 0, 0, 1, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 1, 0, 0, 0},
                {0, 0, 0, 1, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0},
            },
            {5, 3},
            PAT4_TL,
        },
        {
            {
                {0, 0, 0, 0, 0, 0, 0},
                {1, 0, 0, 0, 0, 0, 0},
                {0, 1, 0, 0, 0, 0, 0},
                {0, 0, 1, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 1, 0, 0},
                {0, 0, 0, 0, 0, 1, 0},
                {0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0},
            },
            {4, 3},
            PAT4_TL,
        },
        {
            {
                {0, 0, 0, 0, 0, 0, 0},
                {1, 0, 0, 0, 0, 0, 0},
                {0, 1, 0, 0, 0, 0, 0},
                {0, 0, 1, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 1, 0},
                {0, 0, 0, 0, 0, 0, 1},
                {0, 0, 0, 0, 0, 0, 0, 1},
                {0, 0, 0, 0, 0, 0, 0},
            },
            {4, 3},
            PAT4_44,
        },
        {{
             {0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 1, 1, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 1, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 1, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0},
         },
         {4, 4},
         PAT4_A33},
        {{
             {0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 2, 1, 1, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 1, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 1, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0},
         },
         {4, 4},
         PAT4_OTHERS},
        {{
             {0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 2, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 1, 0, 0, 0, 0, 0},
             {0, 2, 1, 1, 0, 1, 2, 0, 0},
             {0, 0, 0, 0, 0, 1, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 1, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0},
         },
         {4, 4},
         PAT4_OTHERS},
    };
    for (int i = 0; i < n; i++) {
        log("i = %d", i);
        emph_print(tests[i].board, tests[i].pos);
        int forbid = is_forbidden(tests[i].board, tests[i].pos, 1, true);
        log("got %s, expected %s", pattern4_typename[forbid], pattern4_typename[tests[i].id]);
        if (forbid != tests[i].id) {
            log_e("failed.");
            return 1;
        }
    }
    return 0;
}

static int test_pattern(void)
{
    for (int idx = 0; idx < PATTERN_SIZE; idx++) {
        segment_t seg = segment_decode(idx);
        if (seg.data[WIN_LENGTH - 1] != SELF_POS) continue;
        print_segment(segment_decode(idx));
    }
    return 0;
}

int main()
{
    init();

    int ret = 0;
    log("running test");

    log("test pattern");
    test_pattern();

    log("test forbid");
    ret = test_forbid();
    if (ret) return ret;

    log("tests passed.");
    return 0;
}