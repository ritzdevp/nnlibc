# TODO: Re-write Makefile

CC = gcc

CFLAGS  = -g -Wall

TARGET = hello

all: ${TARGET}

$(TARGET): $(TARGET).c
	$(CC) $(CFLAGS) -o $(TARGET) $(TARGET).c


clean:
	$(RM) $(TARGET)