// author: Cauphenuny
// date: 2024/07/26

#include "util.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static char log_buffer[LOG_BUFFER_SIZE];
static int cur_len;
static const char* prompts[PROMPT_SIZE] = {
    [PROMPT_EMPTY] = "",
    [PROMPT_LOG] = BLUE "[LOG] " RESET,
    [PROMPT_INFO] = GREEN "[INFO] " RESET,
    [PROMPT_NOTE] = PURPLE "[NOTE]" RESET,
    [PROMPT_WARN] = YELLOW "[WARN] " RESET,
    [PROMPT_ERROR] = RED "[ERROR] " RESET,
};

/// @brief flush stored logs
void log_flush()
{
    printf("%s", log_buffer);
    log_buffer[0] = '\0', cur_len = 0;
    fflush(stdout);
}

static int _log_islocked = 0, _log_isdisabled = 0;

void log_disable() { _log_isdisabled = 1; }

void log_enable() { _log_isdisabled = 0; }

bool log_disabled() { return _log_isdisabled; }

void log_lock() { _log_islocked = 1; }

void log_unlock() { _log_islocked = 0; if (cur_len) log_flush(); }

bool log_locked() { return _log_islocked; }

int log_write(int log_level, const char* fmt, ...)
{
    if (_log_isdisabled == 1) return 1;

    if (cur_len < LOG_BUFFER_SIZE)
        cur_len +=
            snprintf(log_buffer + cur_len, LOG_BUFFER_SIZE - cur_len, "%s", prompts[log_level]);
    va_list args;
    va_start(args, fmt);
    if (cur_len < LOG_BUFFER_SIZE)
        cur_len += vsnprintf(log_buffer + cur_len, LOG_BUFFER_SIZE - cur_len, fmt, args);
    va_end(args);
    if (!_log_islocked) {
        log_flush();
        return 1;
    } else {
        if (cur_len < LOG_BUFFER_SIZE)
            cur_len += snprintf(log_buffer + cur_len, LOG_BUFFER_SIZE - cur_len, " | ");
    }
    return cur_len;
}

double get_raw_time(void) {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000.0;
}

/// @brief get a timestamp
double record_time(void) { return get_raw_time(); }

/// @brief get current time (ms) after {start_time}
double get_time(double start_time) { return (get_raw_time() - start_time); }

/// @brief pause until inputting a char, ignores inputs before
/// @return char, as int type for EOF
int pause()
{
    const double tim = record_time();
    const int ch = getchar();
    if (ch == EOF) exit(1);
    if (get_time(tim) < 10) return pause();
    return ch;
}

/// @brief get basename of a filename with path
const char* base_name(const char* fullname)
{
    const char* pos = fullname;
    char ch;
#if defined(_WIN32) || defined(_WIN64)
    ch = '\\';
#else
    ch = '/';
#endif
    const int len = strlen(fullname);
    for (int i = 0; i < len; i++) {
        if (fullname[i] == ch) {
            pos = fullname + i + 1;
        }
    }
    return pos;
}

bool file_exists(const char* filename)
{
    FILE* file = fopen(filename, "r");
    if (file) {
        fclose(file);
        return true;
    }
    return false;
}