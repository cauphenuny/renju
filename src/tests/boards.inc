#define TESTS 8
    struct {
        const char* str;
        pattern4_t id;
    } tests[TESTS] = {
        {""                  //
         ". . . . . . . \\"  //
         ". . o . . . . \\"  //
         ". . . o . . . \\"  //
         ". . o o ? . . \\"  //
         ". . . . . . . \\"  //
         ". . . . . . . \\"  //
         ,
         PAT4_A33},
        {""                  //
         ". . . . . . . \\"  //
         ". . o . . . . \\"  //
         ". . . o . . . \\"  //
         ". . o o ? x . \\"  //
         ". . . . . . . \\"  //
         ". . . . . . . \\"  //
         ,
         PAT4_OTHERS},
        {""                  //
         ". . . . . . . \\"  //
         ". . . o . . . \\"  //
         ". . . o . . . \\"  //
         ". . . ? . . . \\"  //
         ". . . o . . . \\"  //
         ". . . o . . . \\"  //
         ". . . o . . . \\"  //
         ,
         PAT4_TL},
        {""  //
         ". . . . . . . . \\"
         ". . . . . x . . \\"
         ". o . . . o . . \\"
         ". . o . . o . . \\"
         ". . . ? o . o . \\"
         ". . . . . o . . \\",
         PAT4_OTHERS},
        {""  //
         ". . . . . . . . . . \\"
         "o . . . . . . . . . \\"
         ". o . . . . . . . . \\"
         ". . o . . . . . . . \\"
         ". . . . . . . . . . \\"
         ". . . . ? . . . . . \\"
         ". . . . . o . . . . \\"
         ". . . . . . o . . . \\"
         ". . . . . . . o . . \\",
         PAT4_OTHERS},
        {""  //
         ". . . . . . . . . . \\"
         ". . . . . . . . . . \\"
         ". . . . . x . . . . \\"
         ". . . x x o . x . . \\"
         ". . x o o o o x . . \\"
         ". . . . o o x . . . \\"
         ". . . ? o o . . . . \\"
         ". . . o x x x . . . \\"
         ". . x . o . . . . . \\",
         PAT4_A33},
        {""  //
         ". . . . . . . . . . \\"
         ". . . . . . . . x . \\"
         ". . . . . . . o . . \\"
         ". . . . . . o . . . \\"
         ". . . . . . . . . . \\"
         ". . . . o . . . . . \\"
         ". . . ? . . . . . . \\"
         ". . . . . . . . . . \\"
         ". o . . . . . . . . \\"
         "o . . . . . . . . . \\",
         PAT4_44},
        {""  //
         ". . . . . . . . . . . \\"
         ". . . . . x . . . . . \\"
         ". . . . . o . . . . . \\"
         ". . o . . o . . . . . \\"
         ". . . o . x o o . . . \\"
         ". . . . o o o . . . . \\"
         ". o o o o ? x . . . . \\"
         ". . . . o x . . . . . \\"
         ". . . . . . . . . . . \\"
         ". . . . . . . . . . . \\"
         ". . . . . . . . . . . \\",
         PAT4_OTHERS},
    };