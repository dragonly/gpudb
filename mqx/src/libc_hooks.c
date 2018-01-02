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
// TODO: Here we only hook the libc malloc, calloc, and free functions,
// assuming user programs only use these interfaces to allocate and release
// host memory. In reality, of course, other functions, such as memalign,
// mmap, and realloc, are also frequently used. Hooks to these interfaces
// need to be added in the future.
#include "libc_hooks.h"

#include "common.h"
#include "cow.h"
#include "list.h"
#include "spinlock.h"

extern volatile int initialized;  // Defined in interfaces.c.
static __thread int libc_hooks_active = 1;     // Thread-local.

struct malloc_area {
    void *addr;
    size_t size;
    struct list_head entry_area;
};

// TODO: This list should be sorted (e.g., using rb tree) to facilitate
// inserting and searching.
static struct list_head list_malloc_areas = LIST_HEAD_INIT(list_malloc_areas);
static struct spinlock lock_malloc_areas = SPINLOCK_INIT;

void add_malloc_area(void *addr, size_t size)
{
    struct malloc_area *new_area =
            (struct malloc_area *)__libc_malloc(sizeof(struct malloc_area));
    CHECK(new_area != NULL, "Failed to allocate a new malloc area struct");
    new_area->addr = addr;
    new_area->size = size;
    acquire(&lock_malloc_areas);
    list_add(&new_area->entry_area, &list_malloc_areas);
    release(&lock_malloc_areas);
}

int del_malloc_area(const void *addr, size_t *size)
{
    struct malloc_area *area;
    struct list_head *pos;

    acquire(&lock_malloc_areas);
    list_for_each(pos, &list_malloc_areas) {
        area = list_entry(pos, struct malloc_area, entry_area);
        if (area->addr == addr) {
            list_del(pos);
            release(&lock_malloc_areas);
            *size = area->size;
            __libc_free(area);
            return 0;
        }
    }
    release(&lock_malloc_areas);

    return -1;
}

MQX_EXPORT
void *malloc(size_t size)
{
    void *addr = NULL;

    if (libc_hooks_active && initialized) {
        libc_hooks_active = 0;
        addr = __libc_malloc(size);
        if (addr && size > 0)
            add_malloc_area(addr, size);
        libc_hooks_active = 1;
    }
    else
        addr = __libc_malloc(size);

    return addr;
}

// TODO: Hooking calloc will cause nv_cudaStreamCreate in
// dma_channel_init to seg fault. This is probably because
// NV's cudaStreaCreate uses calloc function and the calloc
// function has been hooked by NV.
//MQX_EXPORT
//void *calloc(size_t n, size_t size)
//{
//    void *addr = NULL;
//
//    if (libc_hooks_active && initialized) {
//        libc_hooks_active = 0;
//        mqx_print(DEBUG, "In calloc");
//        addr = __libc_calloc(n, size);
//        if (addr && n > 0 && size > 0)
//            add_malloc_area(addr, n * size);
//        libc_hooks_active = 1;
//    }
//    else
//        addr = __libc_malloc(size);
//
//    return addr;
//}

MQX_EXPORT
void free(void *ptr)
{
    size_t size = 0;

    if (libc_hooks_active && initialized) {
        libc_hooks_active = 0;
        if (del_malloc_area(ptr, &size) == -1)
            __libc_free(ptr);
        else
            mqx_libc_free(ptr, size);
        libc_hooks_active = 1;
    }
    else
        __libc_free(ptr);
}

void activate_libc_hooks()
{
    libc_hooks_active = 1;
}

void deactivate_libc_hooks()
{
    libc_hooks_active = 0;
}
