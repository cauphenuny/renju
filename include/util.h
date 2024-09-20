#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>

#define min(x, y) ((x) < (y) ? (x) : (y))
#define max(x, y) ((x) > (y) ? (x) : (y))
#define chkmin(x, y) (x) = min(x, y)
#define chkmax(x, y) (x) = max(x, y)

#define debugf(...) fprintf(stderr, __VA_ARGS__)
/* output style: 16 color mode */
#define CLI_STYLE_NORMAL     "0"
#define CLI_STYLE_BOLD       "1"
#define CLI_STYLE_DARK       "2"
#define CLI_STYLE_BACKGROUND "3"
#define CLI_STYLE_UNDERLINE  "4"

/* output color: 16 color mode */
#define CLI_COLOR_BLACK    "30"
#define CLI_COLOR_RED      "31"
#define CLI_COLOR_GREEN    "32"
#define CLI_COLOR_YELLOW   "33"
#define CLI_COLOR_BLUE     "34"
#define CLI_COLOR_PURPLE   "35"
#define CLI_COLOR_SKYBLUE  "36"
#define CLI_COLOR_WHITE    "37"
#define CLI_COLOR_NORMAL   "38"

#define BLACK                "\e[0;30m"
#define L_BLACK              "\e[1;30m"
#define RED                  "\e[0;31m"
#define L_RED                "\e[1;31m"
#define GREEN                "\e[0;32m"
#define L_GREEN              "\e[1;32m"
#define BROWN                "\e[0;33m"
#define YELLOW               "\e[1;33m"
#define BLUE                 "\e[0;34m"
#define L_BLUE               "\e[1;34m"
#define PURPLE               "\e[0;35m"
#define L_PURPLE             "\e[1;35m"
#define CYAN                 "\e[0;36m"
#define L_CYAN               "\e[1;36m"
#define GRAY                 "\e[0;37m"
#define WHITE                "\e[1;37m"
#define BOLD                 "\e[1m"
#define UNDERLINE            "\e[4m"
#define BLINK                "\e[5m"
#define REVERSE              "\e[7m"
#define HIDE                 "\e[8m"
#define CLEAR                "\e[2J"
#define CLRLINE              "\r\e[K"
#define NONE                 "\e[0m"

const char* basename(const char*);

#ifdef NO_LOG

#    define log_l(...) 1
#    define log        log_l

#    define log_s(...) 1
#    define log_i(...) 1
#    define log_w(...) 1
#    define log_e(...) 1

#else

// log-named log

#    define log_l(fmt, ...)                                                  \
        fprintf(stderr,                                                      \
                "\033[" CLI_STYLE_NORMAL ";" CLI_COLOR_BLUE "m[LOG] \033[0m" \
                "\033[" CLI_STYLE_DARK ";" CLI_COLOR_NORMAL                  \
                "m%s/%s/%d: \033[0m" fmt "\n",                               \
                basename(__FILE__), __func__, __LINE__, ##__VA_ARGS__)

#    define log log_l

// simplified log

#    define log_s(...) fprintf(stderr, __VA_ARGS__)

// info-named log
#    define log_i(fmt, ...)                                                    \
        fprintf(stderr,                                                        \
                "\033[" CLI_STYLE_NORMAL ";" CLI_COLOR_GREEN "m[INFO] \033[0m" \
                "\033[" CLI_STYLE_DARK ";" CLI_COLOR_NORMAL                    \
                "m%s/%s/%d: \033[0m" fmt "\n",                                 \
                basename(__FILE__), __func__, __LINE__, ##__VA_ARGS__)

// warning-named log
#    define log_w(fmt, ...)                                   \
        fprintf(stderr,                                       \
                "\033[" CLI_STYLE_NORMAL ";" CLI_COLOR_YELLOW \
                "m[WARN] \033[0m"                             \
                "\033[" CLI_STYLE_DARK ";" CLI_COLOR_NORMAL   \
                "m%s/%s/%d: \033[0m" fmt "\n",                \
                basename(__FILE__), __func__, __LINE__, ##__VA_ARGS__)

// error-named log
#    define log_e(fmt, ...)                                                   \
        fprintf(stderr,                                                       \
                "\033[" CLI_STYLE_NORMAL ";" CLI_COLOR_RED "m[ERROR] \033[0m" \
                "\033[" CLI_STYLE_DARK ";" CLI_COLOR_NORMAL                   \
                "m%s/%s/%d: \033[0m" fmt "\n",                                \
                basename(__FILE__), __func__, __LINE__, ##__VA_ARGS__)

#endif

#define prompt_getch() fprintf(stderr, "(%s) ", __func__), getchar()
#define prompt() fprintf(stderr, "(%s) ", __func__)

void reset_time(void);
int get_time(void);

#endif