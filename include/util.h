#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>

#define min(x, y) ((x) < (y) ? (x) : (y))
#define max(x, y) ((x) > (y) ? (x) : (y))
#define chkmin(x, y) (x = min(x, y))
#define chkmax(x, y) (x = max(x, y))

#ifndef NOCOLOR
#define BLACK                "\e[0;30m"
#define L_BLACK              "\e[1;30m"
#define RED                  "\e[0;31m"
#define L_RED                "\e[1;31m"
#define GREEN                "\e[0;32m"
#define L_GREEN              "\e[1;32m"
#define YELLOW               "\e[0;33m"
#define L_YELLOW             "\e[1;33m"
#define BLUE                 "\e[0;34m"
#define L_BLUE               "\e[1;34m"
#define PURPLE               "\e[0;35m"
#define L_PURPLE             "\e[1;35m"
#define CYAN                 "\e[0;36m"
#define L_CYAN               "\e[1;36m"
#define GRAY                 "\e[0;37m"
#define WHITE                "\e[1;37m"
#define BOLD                 "\e[1m"
#define DARK                 "\e[2m"
#define UNDERLINE            "\e[4m"
#define BLINK                "\e[5m"
#define REVERSE              "\e[7m"
#define HIDE                 "\e[8m"
#define CLEAR                "\e[2J"
#define CLRLINE              "\r\e[K"
#define NONE                 "\e[0m"
#    else
#define BLACK     "" 
#define L_BLACK   "" 
#define RED       "" 
#define L_RED     "" 
#define GREEN     "" 
#define L_GREEN   "" 
#define YELLOW    "" 
#define L_YELLOW  "" 
#define BLUE      "" 
#define L_BLUE    "" 
#define PURPLE    "" 
#define L_PURPLE  "" 
#define CYAN      "" 
#define L_CYAN    "" 
#define GRAY      "" 
#define WHITE     "" 
#define BOLD      ""
#define DARK      ""
#define UNDERLINE ""
#define BLINK     ""
#define REVERSE   ""
#define HIDE      ""
#define CLEAR     ""
#define NONE      ""
#endif

void pause();
#define prompt_pause() fprintf(stderr, "(%s) ", __func__), pause()
#define prompt()       fprintf(stderr, "(%s) ", __func__)

const char* basename(const char*);

enum log_prompts {
    PROMPT_EMPTY, 
    PROMPT_LOG,
    PROMPT_INFO,
    PROMPT_WARN,
    PROMPT_ERROR,
};

#define LOG_BUFFER_SIZE 512
void log_flush(void);
int log_write(int level, const char* fmt, ...);

#define log_add(level, fmt, ...) \
    log_write(level, DARK "%s/%s/%d: " NONE fmt, basename(__FILE__), __func__, __LINE__, ##__VA_ARGS__)

#define log(...)   log_add(PROMPT_LOG,   __VA_ARGS__)
#define log_l(...) log_add(PROMPT_LOG,   __VA_ARGS__)
#define log_i(...) log_add(PROMPT_INFO,  __VA_ARGS__)
#define log_w(...) log_add(PROMPT_WARN,  __VA_ARGS__)
#define log_e(...) log_add(PROMPT_ERROR, __VA_ARGS__), prompt_pause()
#define log_s(...) log_write(PROMPT_EMPTY, __VA_ARGS__)

int record_time(void);
int get_time(int);

#endif
