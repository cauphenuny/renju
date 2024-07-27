#pragma once

#include <stdio.h>

#define min(x, y) ((x) < (y) ? (x) : (y))
#define max(x, y) ((x) > (y) ? (x) : (y))
#define chkmin(x, y) (x) = min(x, y)
#define chkmax(x, y) (x) = max(x, y)

#define debugf(...) fprintf(stderr, __VA_ARGS__)
#define debugv(fmt, ...) \
    fprintf(stderr, "<%s:%d> " fmt "\n", __func__, __LINE__, ##__VA_ARGS__)
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

char* basename(char*);

// infomation log

#define logi(fmt, ...) \
    fprintf(stderr, \
    "\033[" CLI_STYLE_NORMAL ";" CLI_COLOR_BLUE "m[LOG] \033[0m" \
    "\033[" CLI_STYLE_DARK ";" CLI_COLOR_NORMAL "m%s/%s/%d: \033[0m" \
    fmt "\n", basename(__FILE__), __func__, __LINE__, ## __VA_ARGS__)

#define log logi

// simplified log

#define logs(...) fprintf(stderr, __VA_ARGS__)

// warning style log
#define logw(fmt, ...) \
    fprintf(stderr, \
    "\033[" CLI_STYLE_NORMAL ";" CLI_COLOR_YELLOW "m[WARN] \033[0m" \
    "\033[" CLI_STYLE_DARK ";" CLI_COLOR_NORMAL "m%s/%s/%d: \033[0m" \
    fmt "\n", basename(__FILE__), __func__, __LINE__, ## __VA_ARGS__)

// error style log
#define loge(fmt, ...) \
    fprintf(stderr, \
    "\033[" CLI_STYLE_NORMAL ";" CLI_COLOR_RED "m[ERROR] \033[0m" \
    "\033[" CLI_STYLE_DARK ";" CLI_COLOR_NORMAL "m%s/%s/%d: \033[0m" \
    fmt "\n", basename(__FILE__), __func__, __LINE__, ## __VA_ARGS__)

void reset_time(void);
int get_time(void);

