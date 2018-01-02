/*
 * Copyright (c) 2014 Kaibo Wang (wkbjerry@gmail.com)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <stdio.h>
#include <stdlib.h>

#include "common.h"
#ifdef MQX_PRINT_BUFFER
#include "libc_hooks.h"  // For __libc_malloc and __libc_free
#endif

char *MQX_PRINT_MSG[PRINT_LEVELS] = {
    "S",
    "F",
    "E",
    "W",
    "I",
    "D",
};

static void show_stackframe() {
  char **messages = (char **)NULL;
  int i, trace_size = 0;
  void *trace[32];

  trace_size = backtrace(trace, 32);
  messages = backtrace_symbols(trace, trace_size);
  mqx_print(INFO, "Printing stack frames:");
  for (i=0; i < trace_size; ++i)
      mqx_print(INFO, "\t%s", messages[i]);
}

void panic(const char *msg)
{
    mqx_print(FATAL, "%s", msg);
    show_stackframe();
    exit(-1);
}

#ifdef MQX_PRINT_BUFFER
char *mqx_print_buffer = NULL;
#define PRINT_BUFFER_SIZE        (128L * 1024L * 1024L)
int mqx_print_lines = 0;
int mqx_print_head = 0;
struct spinlock mqx_print_lock;
#endif

void mqx_print_init()
{
#ifdef MQX_PRINT_BUFFER
    int i;
    initlock(&mqx_print_lock);
    mqx_print_buffer = (char *)__libc_malloc(PRINT_BUFFER_SIZE);
    if (!mqx_print_buffer) {
        mqx_print(FATAL, "Failed to initialize mqx_print buffer");
        exit(-1);
    }
    for (i = 0; i < PRINT_BUFFER_SIZE; i += 4096)
        mqx_print_buffer[i] = 'x';
#endif
}

void mqx_print_fini()
{
#ifdef MQX_PRINT_BUFFER
    int i, head = 0, len;
    if (mqx_print_buffer) {
        for (i = 0; i < mqx_print_lines; i++) {
            len = printf("%s", mqx_print_buffer + head);
            head += len + 1;
        }
        __libc_free(mqx_print_buffer);
        mqx_print_buffer = NULL;
    }
#endif
}
