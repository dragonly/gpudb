/*
 * Copyright (c) 2018 Yilong Li <liyilongko@gmail.com>
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
#ifndef _MPS_STATS_H_
#define _MPS_STATS_H_

#include <sys/time.h>

#define TVALMS(t) ((t).tv_sec * 1000.0 + (t).tv_usec / 1000.0)
#define TDIFFMS(t1, t2) (TVALMS(t2) - TVALMS(t1))
#define TVALUS(t) ((t).tv_sec * 1000000.0 + (t).tv_usec)
#define TDIFFUS(t1, t2) (TVALUS(t2) - TVALUS(t1))

#ifdef MPS_CONFIG_STATS
// There can be no break statement or variable declaration in
// the program segment measured by the time begin & end macros.
#define stats_time_begin()  \
  struct timeval _t1, _t2;  \
  gettimeofday(&_t1, NULL)

#define stats_time_end(RESULT) \
    gettimeofday(&_t2, NULL);   \
    RESULT = TDIFFMS(_t1, _t2);

#else

#define stats_time_begin()
#define stats_time_end(RESULT)

#endif

struct mps_stats {
  pthread_mutex_t mutex;
  double time_load_region;
  double time_load_region_pta;
  double time_load_region_memset;
  double time_sync_block;
  double time_syncStoD;
  double time_syncDtoS;
  double time_syncDtoS1;
  double time_memcpyDtoD;
};

#endif
