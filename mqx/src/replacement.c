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
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "client.h"
#include "common.h"
#include "libc_hooks.h"
#include "protocol.h"
#include "replacement.h"

extern struct global_context *global_ctx;

static long compute_cost(struct region_elem *pelem, double lru_pos) {
  const double f = 0.01;
  return pelem->cost_evict + (long)(f * pelem->size * lru_pos);
}

int victim_select_cost(long size_needed, struct region **excls, int nexcl, struct list_head *victims) {
  struct region_list *prgns = &global_ctx->regions;
  struct region_elem *pelem;
  struct region *r;
  struct victim *v;
  long cost, cost_min = LONG_MAX;
  void *addr = NULL;
  int cid = -1, irgn;
  int lru_pos = 1;

  v = (struct victim *)__libc_malloc(sizeof(*v));
  if (!v) {
    mqx_print(FATAL, "Evict: failed to allocate memory for victim: %s", strerror(errno));
    return -1;
  }
  v->client = -1;
  v->r = NULL;

  acquire(&prgns->lock);
  for (irgn = prgns->ilru; irgn != -1; irgn = prgns->rgns[irgn].iprev) {
    pelem = &prgns->rgns[irgn];
    if (atomic_read(&pelem->pinned) == 0 &&
        (!is_client_local(pelem->cid) || !is_included((const void **)excls, nexcl, pelem->addr))) {
      cost = compute_cost(pelem, lru_pos / (double)prgns->nrgns);
      if (cost < cost_min) {
        cost_min = cost;
        addr = pelem->addr;
        cid = pelem->cid;
      }
      ++lru_pos;
    }
  }

  if (addr != NULL) {
    if (is_client_local(cid)) {
      release(&prgns->lock);
      r = (struct region *)(addr);
      if (try_acquire(&r->lock)) {
        if (r->state == STATE_ATTACHED && !region_pinned(r)) {
          r->state = STATE_EVICTING;
          release(&r->lock);
          v->r = r;
          v->client = -1;
          v->addr = NULL;
          list_add(&v->entry, victims);
        } else
          release(&r->lock);
      }
    } else {
      global_ctx->clients[cid].pinned++;
      release(&prgns->lock);
      v->r = NULL;
      v->client = cid;
      v->addr = addr;
      list_add(&v->entry, victims);
    }
  } else
    release(&prgns->lock);

  if (v->client == -1 && v->r == NULL) {
    mqx_print(DEBUG, "Evict: no victim selected");
    __libc_free(v);
    return 1;
  }
  return 0;
}
