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
#ifndef _MQX_ATOMIC_H_
#define _MQX_ATOMIC_H_

typedef volatile int atomic_t;
typedef volatile long latomic_t;

static inline void atomic_set(atomic_t *ptr, int val) { *ptr = val; }

static inline int atomic_read(atomic_t *ptr) { return *ptr; }

static inline int atomic_add(atomic_t *ptr, int val) { return __sync_fetch_and_add(ptr, val); }

static inline int atomic_inc(atomic_t *ptr) { return __sync_fetch_and_add(ptr, 1); }

static inline int atomic_sub(atomic_t *ptr, int val) { return __sync_fetch_and_sub(ptr, val); }

static inline int atomic_dec(atomic_t *ptr) { return __sync_fetch_and_sub(ptr, 1); }

static inline void atomic_setl(latomic_t *ptr, long val) { *ptr = val; }

static inline int atomic_readl(latomic_t *ptr) { return *ptr; }

static inline long atomic_addl(latomic_t *ptr, long val) { return __sync_fetch_and_add(ptr, val); }

static inline long atomic_subl(latomic_t *ptr, long val) { return __sync_fetch_and_sub(ptr, val); }

static inline long atomic_incl(latomic_t *ptr) { return __sync_fetch_and_add(ptr, 1L); }

static inline long atomic_decl(latomic_t *ptr) { return __sync_fetch_and_sub(ptr, 1L); }

#endif
