src = $(wildcard *.c)
head = $(wildcard *.h)
obj = $(src:%.c=%.o)
target = gomoku3
CFLAGS = -Wall -Wextra -Wshadow -O2
#LDFLAGS = -g -fsanitize=undefined,address,leak,null,bounds
LDFLAGS = -lm

%.o: %.c $(head) Makefile 
	$(CC) -c $< -o $@ $(CFLAGS) $(CPPFLAGS)

$(target): $(obj)
	$(CC) $(obj) -o $(target) $(LDFLAGS)

.PHONY: clean rebuild run all

clean:
	rm -rf $(obj) $(target) 

rebuild: clean $(target)

run: $(target)
	./$(target)

all: $(target)
