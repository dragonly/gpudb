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
#include <errno.h>
#include <pthread.h>
#include <semaphore.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h> // For strerror
#include <sys/mman.h>
#include <unistd.h>

#include "common.h"
#include "core.h"
#include "libc_hooks.h"
#include "list.h"
#include "spinlock.h"

struct sigaction old_sigsegv_act;

#define MAX_REGIONS_PER_COW_RANGE 4

struct cow_range {
  void *addr_begin;
  void *addr_end;
  struct region *r[MAX_REGIONS_PER_COW_RANGE];
  int n_r;
  struct list_head entry_cow;
};

// List of COW address ranges.
// TODO: Currently works, but not thread-safe.
static struct list_head list_cow_ranges = LIST_HEAD_INIT(list_cow_ranges);

// For syncing the COW signal handler and the COW worker thread.
static sem_t sem_cow; // For informing the worker thread of COW work.
static int pfd_read;  // Pipe fds for informing the COW signal
static int pfd_write; // handler of the finishing of COW work.

// The COW address range that needs to be copied by the cow_worker thread.
static struct cow_range *range_copy = NULL;

// COW worker thread ID.
static pthread_t tid_cow_worker;

int wait_for_cow_work() {
again:
  if (sem_wait(&sem_cow) == -1) {
    if (errno == EINTR)
      goto again;
    else
      return -1;
  }
  return 0;
}

void do_cow_work(struct cow_range *range) {
  int i;

  for (i = 0; i < range->n_r; ++i) {
    mqx_handle_cow(range->addr_begin, range->addr_end, range->r[i]);
  }
  mprotect(range->addr_begin, (unsigned long)(range->addr_end - range->addr_begin), PROT_READ | PROT_WRITE);
  __libc_free(range);
}

int cow_work_done() {
  char dummy_write = 'x';

again:
  if (write(pfd_write, &dummy_write, 1) < 0) {
    if (errno == EINTR)
      goto again;
    return -1;
  }
  return 0;
}

// The thread that handles COW copying from user source buffer to swap buffer.
void *thread_cow_worker(void *arg_unused) {
  struct cow_range *range;

  deactivate_libc_hooks();

  while (1) {
    if (wait_for_cow_work() == -1)
      break;
    range = range_copy;
    range_copy = NULL;
    CHECK(range != NULL, "cow_worker: COW range cannot be null");
    do_cow_work(range);
    if (cow_work_done() == -1)
      break;
  }

  return NULL;
}

int wait_for_cow_work_to_finish() {
  char dummy_read;

again:
  if (read(pfd_read, &dummy_read, 1) < 0) {
    if (errno == EINTR)
      goto again;
    return -1;
  }
  return 0;
}

static void cow_handler(int sig, siginfo_t *si, void *unused) {
  struct cow_range *range;
  struct list_head *pos;

  if (si->si_code != SEGV_ACCERR)
    goto default_handler;

  list_for_each(pos, &list_cow_ranges) {
    range = list_entry(pos, struct cow_range, entry_cow);
    if (si->si_addr >= range->addr_begin && si->si_addr < range->addr_end) {
      list_del(pos);
      if (range_copy != NULL)
        abort();
      range_copy = range;
      sem_post(&sem_cow); // Inform worker thread of the incoming work.
      if (wait_for_cow_work_to_finish() == -1)
        goto default_handler;
      return;
    }
  }

default_handler:
  // Invoke the default SIGSEGV handler.
  if (sigaction(SIGSEGV, &old_sigsegv_act, NULL) == -1)
    abort();
  // mqx_print(DEBUG, "Default SIGSEGV handler will be invoked");
  raise(SIGSEGV);
}

int cow_init() {
  struct sigaction sa;
  int pipefd[2];

  if (sem_init(&sem_cow, 0, 0) == -1) {
    mqx_print(FATAL, "Failed to initialize COW semaphore: %s.", strerror(errno));
    return -1;
  }
  if (pipe(pipefd) == -1) {
    mqx_print(FATAL, "Failed to create pipe for COW handling: %s.", strerror(errno));
    sem_destroy(&sem_cow);
    return -1;
  }
  pfd_read = pipefd[0];
  pfd_write = pipefd[1];
  if (pthread_create(&tid_cow_worker, NULL, thread_cow_worker, NULL) != 0) {
    mqx_print(FATAL, "Failed to create COW worker thread: %s.", strerror(errno));
    close(pfd_read);
    close(pfd_write);
    sem_destroy(&sem_cow);
    return -1;
  }

  sa.sa_sigaction = cow_handler;
  sa.sa_flags = SA_SIGINFO;
  sigemptyset(&sa.sa_mask);
  if (sigaction(SIGSEGV, &sa, &old_sigsegv_act) == -1) {
    mqx_print(FATAL, "Failed to register SIGSEGV signal handler: %s.", strerror(errno));
    pthread_cancel(tid_cow_worker);
    pthread_join(tid_cow_worker, NULL);
    close(pfd_read);
    close(pfd_write);
    sem_destroy(&sem_cow);
    return -1;
  }

  return 0;
}

void cow_fini() {
  sigaction(SIGSEGV, &old_sigsegv_act, NULL);
  // Clean COW ranges
  while (!list_empty(&list_cow_ranges)) {
    struct list_head *p = list_cow_ranges.next;
    struct cow_range *range = list_entry(p, struct cow_range, entry_cow);
    list_del(p);
    mprotect(range->addr_begin, (unsigned long)(range->addr_end - range->addr_begin), PROT_READ | PROT_WRITE);
    __libc_free(range);
  }
  pthread_cancel(tid_cow_worker);
  pthread_join(tid_cow_worker, NULL);
  close(pfd_read);
  close(pfd_write);
  sem_destroy(&sem_cow);
}

// The return value tells whether the new cow_range has been added
// to the cow_range list.
static int fix_cow_range(struct cow_range *range_fix, void *new_addr, unsigned long new_bytes,
                         struct region *new_region) {
  // If the new cow_range is exactly the same with an existing
  // range, add the new cow_range to the existing entry.
  if (range_fix->addr_begin == new_addr && range_fix->addr_end == new_addr + new_bytes) {
    if (range_fix->n_r == MAX_REGIONS_PER_COW_RANGE) {
      // If the number of regions corresponding to this
      // cow_range has reached the max, we need to break
      // one and then add the new region to the entry.
      struct region *old_region = range_fix->r[--(range_fix->n_r)];
      mqx_handle_cow(new_addr, new_addr + new_bytes, old_region);
    }
    range_fix->r[range_fix->n_r++] = new_region;
    return 1;
  }

  // Otherwise, we need to delete the existing cow_range, doing
  // all lazy data copies.
  list_del(&range_fix->entry_cow);
  while (range_fix->n_r-- > 0) {
    struct region *old_region = range_fix->r[range_fix->n_r];
    mqx_handle_cow(range_fix->addr_begin, range_fix->addr_end, old_region);
  }
  mprotect(range_fix->addr_begin, (unsigned long)(range_fix->addr_end - range_fix->addr_begin), PROT_READ | PROT_WRITE);
  __libc_free(range_fix);
  return 0;
}

int add_cow_range(void *new_addr, unsigned long new_bytes, struct region *new_region) {
  struct cow_range *range, *range_fix = NULL;
  struct list_head *pos;

  CHECK(!((unsigned long)new_addr & PAGEMASK), "COW address must be page-aligned");
  CHECK(new_bytes > 0 && !(new_bytes & PAGEMASK), "COW bytes must be multiples of page size");

  mqx_print(DEBUG, "add_cow_range: %p %lu %p", new_addr, new_bytes, new_region);

  // Search for all existing cow ranges, fixing any one that overlaps
  // with the new cow range being added.
  list_for_each(pos, &list_cow_ranges) {
    if (range_fix) {
      if (fix_cow_range(range_fix, new_addr, new_bytes, new_region))
        return 0;
      range_fix = NULL;
    }
    range = list_entry(pos, struct cow_range, entry_cow);
    // If an existing cow range overlaps with the new one
    if (new_addr < range->addr_end && new_addr + new_bytes > range->addr_begin) {
      range_fix = range;
    }
  }
  if (range_fix) {
    if (fix_cow_range(range_fix, new_addr, new_bytes, new_region))
      return 0;
    range_fix = NULL;
  }

  // A new cow_range entry needs to be added.
  if (mprotect(new_addr, new_bytes, PROT_READ) == -1)
    return -1;
  range = __libc_malloc(sizeof(struct cow_range));
  range->addr_begin = new_addr;
  range->addr_end = new_addr + new_bytes;
  range->r[0] = new_region;
  range->n_r = 1;
  list_add(&range->entry_cow, &list_cow_ranges);

  return 0;
}

int del_cow_range(void *addr, unsigned long bytes, struct region *r) {
  struct cow_range *range;
  struct list_head *pos;
  int found = 0;

  CHECK(!((unsigned long)addr & PAGEMASK), "COW address must be page-aligned");
  CHECK(bytes > 0 && !(bytes & PAGEMASK), "COW bytes must be multiples of page size");

  list_for_each(pos, &list_cow_ranges) {
    range = list_entry(pos, struct cow_range, entry_cow);
    if (range->addr_begin == addr && range->addr_end == addr + bytes) {
      found = 1;
      break;
    }
  }

  if (found) {
    int i;
    for (i = 0; i < range->n_r; ++i) {
      if (range->r[i] == r) {
        break;
      }
    }
    CHECK(i < range->n_r, "region must be registered in cow_range");
    if (i < range->n_r - 1)
      range->r[i] = range->r[range->n_r - 1];
    range->n_r--;
    if (range->n_r == 0) {
      list_del(pos);
      __libc_free(range);
      mqx_print(DEBUG, "errno(%d)", errno);
      int ret = mprotect(addr, bytes, PROT_READ | PROT_WRITE);
      mqx_print(DEBUG, "ret(%d) errno(%d) addr(%p) size(%ld)", ret, errno, addr, bytes);
      return ret;
    }
    return 0;
  } else {
    mqx_print(WARN, "Cannot find cow range");
    return -1;
  }
}

void mqx_libc_free(void *ptr, size_t size) {
  struct cow_range *range;
  struct list_head *pos;
  int i;

  // Traverse through all cow ranges, and handle each cow range
  // that overlaps with the buffer area being freed.
  list_for_each(pos, &list_cow_ranges) {
    range = list_entry(pos, struct cow_range, entry_cow);
    if (ptr < range->addr_end && ptr + size > range->addr_begin) {
      list_del(pos);
      for (i = 0; i < range->n_r; ++i) {
        mqx_print(DEBUG, "Freeing COW source buffer %p %lu %p %p", ptr, size, range->addr_begin, range->addr_end);
        mqx_handle_cow(range->addr_begin, range->addr_end, range->r[i]);
      }
      mprotect(range->addr_begin, (unsigned long)(range->addr_end - range->addr_begin), PROT_READ | PROT_WRITE);
      __libc_free(range);
      break;
    }
  }
  __libc_free(ptr);
}
