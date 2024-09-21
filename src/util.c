// author: Cauphenuny <https://cauphenuny.github.io/>
// date: 2024/07/26
#include <time.h>
#include <string.h>
#include "util.h"

#ifndef ECHO_LOG

char log_buffer[LOG_BUFFER_SIZE];

void log_flush()
{
    printf("%s\n", log_buffer);
    log_buffer[0] = '\0';
}

#endif

long long get_raw_time(void) {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    long long tim = time.tv_sec * 1000 + time.tv_nsec / 1000000;
    return tim;
}

int record_time(void) {
    return get_raw_time();
}

int get_time(int start_time) {
    return (int)(get_raw_time() - start_time);
}

#ifdef ECHO_LOG

const char* basename(const char* fullname) {
    const char* pos = fullname;
    char ch;
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

#endif