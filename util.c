// author: Cauphenuny <https://cauphenuny.github.io/>
// date: 2024/07/26
#include <time.h>

long long tim_;

long long get_raw_time() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    long long tim = time.tv_sec * 1000 + time.tv_nsec / 1000000;
    return tim;
}

void reset_time() {
    tim_ = get_raw_time();
}

int get_time() {
    return (int)(get_raw_time() - tim_);
}
