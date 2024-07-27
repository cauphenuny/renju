// author: Cauphenuny <https://cauphenuny.github.io/>
// date: 2024/07/26
#include <time.h>
#include <string.h>
#include "util.h"

long long tim_;

long long get_raw_time(void) {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    long long tim = time.tv_sec * 1000 + time.tv_nsec / 1000000;
    return tim;
}

void reset_time(void) {
    tim_ = get_raw_time();
}

int get_time(void) {
    return (int)(get_raw_time() - tim_);
}


char* basename(char* fullname) {
    char* pos = fullname, ch;
#if defined(_WIN32) || defined(_WIN64)
    ch = '\\';
#else
    ch = '/';
#endif
    int len = strlen(fullname);
    for (int i = 0; i < len; i++) {
        if (fullname[i] == ch) {
            pos = fullname + i + 1;
        }
    }
    return pos;
}
