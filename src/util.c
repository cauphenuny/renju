// author: Cauphenuny
// date: 2024/07/26
#include "util.h"

#include <stdarg.h>
#include <string.h>
#include <time.h>

static char log_buffer[LOG_BUFFER_SIZE];
static int cur_len;
static const char* prompts[] = {
    [PROMPT_EMPTY] = "",
    [PROMPT_LOG] = BLUE "[LOG] " NONE,
    [PROMPT_INFO] = GREEN "[INFO] " NONE,
    [PROMPT_WARN] = YELLOW "[WARN] " NONE,
    [PROMPT_ERROR] = RED "[ERROR] " NONE,
};

void log_flush()
{
    printf("%s\n", log_buffer);
    log_buffer[0] = '\0', cur_len = 0;
}

int log_add(int log_level, const char* fmt, ...)
{
#ifdef DISABLE_LOG
    return 1;
#endif
    cur_len += snprintf(log_buffer + cur_len, LOG_BUFFER_SIZE - cur_len, "%s", prompts[log_level]);
    va_list args;
    va_start(args, fmt);
    cur_len += vsnprintf(log_buffer + cur_len, LOG_BUFFER_SIZE - cur_len, fmt, args);
    va_end(args);
#ifdef INSTANT_LOG
    log_flush();
#else
    cur_len += snprintf(log_buffer + cur_len, LOG_BUFFER_SIZE - cur_len, " | ");
#endif
    return cur_len;
}

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

void pause()
{
    int tim = record_time();
    char c = getchar();
    if (get_time(tim) < 10) return pause();
}

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