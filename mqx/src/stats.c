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
#include <string.h>

#include "common.h"
#include "stats.h"

#ifdef MQX_CONFIG_STATS
// Current # of bytes of active device memory
long bytes_mem_active = 0L;
#endif

void stats_init(struct statistics *pstats)
{
#ifdef MQX_CONFIG_STATS
    memset(pstats, 0, sizeof(*pstats));
#endif
}

void stats_print(struct statistics *pstats)
{
#ifdef MQX_CONFIG_STATS
    acquire(&pstats->lock);
    mqx_print(STAT, "------------------------------");
    mqx_print(STAT, "bytes_mem_alloc     %ld", pstats->bytes_mem_alloc);
    mqx_print(STAT, "bytes_mem_peak      %ld", pstats->bytes_mem_peak);
    mqx_print(STAT, "bytes_mem_freed     %ld", pstats->bytes_mem_freed);
    mqx_print(STAT, "bytes_htod          %ld", pstats->bytes_htod);
    mqx_print(STAT, "time_htod           %.3lf", pstats->time_htod);
    mqx_print(STAT, "bytes_htod_cow      %ld", pstats->bytes_htod_cow);
    mqx_print(STAT, "time_htod_cow       %.3lf", pstats->time_htod_cow);
    mqx_print(STAT, "bytes_dtoh          %ld", pstats->bytes_dtoh);
    mqx_print(STAT, "time_dtoh           %.3lf", pstats->time_dtoh);
    mqx_print(STAT, "bytes_dtod          %ld", pstats->bytes_dtod);
    mqx_print(STAT, "time_dtod           %.3lf", pstats->time_dtod);
    mqx_print(STAT, "bytes_memset        %ld", pstats->bytes_memset);
    mqx_print(STAT, "time_memset         %.3lf", pstats->time_memset);
    mqx_print(STAT, "------------------------------");
    mqx_print(STAT, "bytes_u2s           %ld", pstats->bytes_u2s);
    mqx_print(STAT, "time_u2s            %.3lf", pstats->time_u2s);
    mqx_print(STAT, "bytes_s2d           %ld", pstats->bytes_s2d);
    mqx_print(STAT, "time_s2d            %.3lf", pstats->time_s2d);
    mqx_print(STAT, "bytes_u2d           %ld", pstats->bytes_u2d);
    mqx_print(STAT, "time_u2d            %.3lf", pstats->time_u2d);
    mqx_print(STAT, "bytes_d2s           %ld", pstats->bytes_d2s);
    mqx_print(STAT, "time_d2s            %.3lf", pstats->time_d2s);
    mqx_print(STAT, "bytes_s2u           %ld", pstats->bytes_s2u);
    mqx_print(STAT, "time_s2u            %.3lf", pstats->time_s2u);
    mqx_print(STAT, "------------------------------");
    mqx_print(STAT, "bytes_evict_needed  %ld", pstats->bytes_evict_needed);
    mqx_print(STAT, "bytes_evict_space   %ld", pstats->bytes_evict_space);
    mqx_print(STAT, "count_evict_victims %ld", pstats->count_evict_victims);
    mqx_print(STAT, "count_evict_fail    %ld", pstats->count_evict_fail);
    mqx_print(STAT, "count_attach_fail   %ld", pstats->count_attach_fail);
    mqx_print(STAT, "------------------------------");
    mqx_print(STAT, "time_sync_attach    %.3lf", pstats->time_sync_attach);
    mqx_print(STAT, "time_sync_block     %.3lf", pstats->time_sync_block);
    mqx_print(STAT, "time_sync_rw        %.3lf", pstats->time_sync_rw);
    mqx_print(STAT, "time_kernel         %.3lf", pstats->time_kernel);
    mqx_print(STAT, "time_attach         %.3lf", pstats->time_attach);
    mqx_print(STAT, "time_load           %.3lf", pstats->time_load);
    mqx_print(STAT, "count_kernels       %ld", pstats->count_kernels);
    mqx_print(STAT, "------------------------------");
    release(&pstats->lock);
#endif
}
