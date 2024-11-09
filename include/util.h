#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>
#include <stdbool.h>

#define min(x, y) ((x) < (y) ? (x) : (y))
#define max(x, y) ((x) > (y) ? (x) : (y))
#define chkmin(x, y) (x = min(x, y))
#define chkmax(x, y) (x = max(x, y))

#ifndef NOCOLOR
#    define BLACK     "\033[0;30m"
#    define L_BLACK   "\033[1;30m"
#    define RED       "\033[0;31m"
#    define L_RED     "\033[1;31m"
#    define GREEN     "\033[0;32m"
#    define L_GREEN   "\033[1;32m"
#    define YELLOW    "\033[0;33m"
#    define L_YELLOW  "\033[1;33m"
#    define BLUE      "\033[0;34m"
#    define L_BLUE    "\033[1;34m"
#    define PURPLE    "\033[0;35m"
#    define L_PURPLE  "\033[1;35m"
#    define CYAN      "\033[0;36m"
#    define L_CYAN    "\033[1;36m"
#    define GRAY      "\033[0;37m"
#    define WHITE     "\033[1;37m"
#    define BOLD      "\033[1m"
#    define DARK      "\033[2m"
#    define UNDERLINE "\033[4m"
#    define BLINK     "\033[5m"
#    define REVERSE   "\033[7m"
#    define HIDE      "\033[8m"
#    define CLEAR     "\033[2J"
#    define CLRLINE   "\r\033[K"
#    define RESET     "\033[0m"

#else
#    define BLACK     ""
#    define L_BLACK   ""
#    define RED       ""
#    define L_RED     ""
#    define GREEN     ""
#    define L_GREEN   ""
#    define YELLOW    ""
#    define L_YELLOW  ""
#    define BLUE      ""
#    define L_BLUE    ""
#    define PURPLE    ""
#    define L_PURPLE  ""
#    define CYAN      ""
#    define L_CYAN    ""
#    define GRAY      ""
#    define WHITE     ""
#    define BOLD      ""
#    define DARK      ""
#    define UNDERLINE ""
#    define BLINK     ""
#    define REVERSE   ""
#    define HIDE      ""
#    define CLEAR     ""
#    define RESET     ""
#endif

char pause();
#define prompt_pause() (fprintf(stderr, "(%s) ", __func__), pause())
#define prompt()       fprintf(stderr, "(%s) ", __func__)

const char* base_name(const char*);

enum log_prompts {
    PROMPT_EMPTY, 
    PROMPT_LOG,
    PROMPT_INFO,
    PROMPT_WARN,
    PROMPT_ERROR,
};

#define LOG_BUFFER_SIZE 512
void log_flush(void);
void log_enable();
void log_disable();
void log_lock();
void log_unlock();
bool log_locked();
bool log_disabled();
int log_write(int level, const char* fmt, ...);

#ifndef BOTZONE
#define log_add(level, fmt, ...)                                                           \
    log_write(level, DARK "%s/%s/%d: " RESET fmt "\n", base_name(__FILE__), __func__, __LINE__, \
              ##__VA_ARGS__)
#else
#define log_add(level, fmt, ...) log_write(level, "%s: " fmt, __func__, ##__VA_ARGS__)
#endif

#define log(...)   log_add(PROMPT_LOG,   __VA_ARGS__)
#define log_l(...) log_add(PROMPT_LOG,   __VA_ARGS__)
#define log_i(...) log_add(PROMPT_INFO,  __VA_ARGS__)
#define log_w(...) log_add(PROMPT_WARN,  __VA_ARGS__)
#define log_e(...) log_add(PROMPT_ERROR, __VA_ARGS__)
#define log_s(...) log_write(PROMPT_EMPTY, __VA_ARGS__)

#define chkscanf(...) { if (scanf(__VA_ARGS__) == EOF) exit(1); }
#define prompt_scanf(...) { prompt(); chkscanf(__VA_ARGS__); }

int record_time(void);
int get_time(int start_time);

#endif
