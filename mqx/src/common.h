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
#ifndef _MQX_COMMON_H_
#define _MQX_COMMON_H_

#include <execinfo.h>
#include <stdio.h>
#include <cuda.h>
#ifdef MQX_CONFIG_PROFILE
#include "stats.h"    // for TVAL macro
#include <sys/time.h> // for gettimeofday
#endif

#define MQX_EXPORT __attribute__((__visibility__("default")))

#define STAT 0
#define FATAL 1
#define ERROR 2
#define WARN 3
#define INFO 4
#define DEBUG 5
#define PRINT_LEVELS 6

// All MQX messages whose print levels are less than or equal to
// MQX_PRINT_LEVEL will be printed.
#ifndef MQX_PRINT_LEVEL
#define MQX_PRINT_LEVEL ERROR
#endif

extern char *MQX_PRINT_MSG[];

#ifdef MQX_PRINT_BUFFER
#include "spinlock.h"
#include <sys/time.h>

extern char *mqx_print_buffer;
extern int mqx_print_lines;
extern int mqx_print_head;
extern struct spinlock mqx_print_lock;
#define mqx_print(lvl, fmt, arg...)                                                                                    \
  do {                                                                                                                 \
    if (lvl <= MQX_PRINT_LEVEL) {                                                                                      \
      int len;                                                                                                         \
      struct timeval t;                                                                                                \
      gettimeofday(&t, NULL);                                                                                          \
      acquire(&mqx_print_lock);                                                                                        \
      len = sprintf(mqx_print_buffer + mqx_print_head, "%s %lf " fmt "\n", MQX_PRINT_MSG[lvl],                         \
                    ((double)(t.tv_sec) + t.tv_usec / 1000000.0), ##arg);                                              \
      mqx_print_lines++;                                                                                               \
      mqx_print_head += len + 1;                                                                                       \
      release(&mqx_print_lock);                                                                                        \
    }                                                                                                                  \
  } while (0)
#else
#define mqx_print(lvl, fmt, arg...)                                                                                    \
  do {                                                                                                                 \
    if (lvl <= MQX_PRINT_LEVEL) {                                                                                      \
      if (lvl > WARN) {                                                                                                \
        fprintf(stdout, "%s " fmt "\n", MQX_PRINT_MSG[lvl], ##arg);                                                    \
      } else {                                                                                                         \
        fprintf(stderr, "%s " fmt "\n", MQX_PRINT_MSG[lvl], ##arg);                                                    \
      }                                                                                                                \
    }                                                                                                                  \
  } while (0)
#endif

void mqx_print_init();
void mqx_print_fini();

#ifdef MQX_CONFIG_PROFILE
#define mqx_profile(fmt, arg...)                                                                                       \
  do {                                                                                                                 \
    struct timeval tprof;                                                                                              \
    gettimeofday(&tprof, NULL);                                                                                        \
    mqx_print(STAT, "PROF %lf " fmt "\n", TVAL(tprof), ##arg);                                                         \
  } while (0)
#else
#define mqx_profile(fmt, arg...)
#endif

#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef gettid
#define _GNU_SOURCE
#include <sys/syscall.h>
#include <unistd.h>
// Gets the id of the caller thread.
static inline pid_t gettid() { return (pid_t)syscall(__NR_gettid); }
#endif

void panic(const char *msg);

#define CHECK(cond, msg)                                                                                               \
  do {                                                                                                                 \
    if (!(cond))                                                                                                       \
      panic(msg);                                                                                                      \
  } while (0)

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
void __checkCudaErrors(CUresult err, const char *file, const int line);
const char *getCudaDrvErrorString(CUresult error_id);

#endif
