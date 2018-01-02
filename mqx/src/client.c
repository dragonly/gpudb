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
#include <fcntl.h>
#include <math.h>
#include <mqueue.h>
#include <pthread.h>
#include <semaphore.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h> /* For mode constants */

#include "advice.h"
#include "client.h"
#include "common.h"
#include "core.h"
#include "msq.h"
#include "protocol.h"

sem_t *sem_launch = SEM_FAILED;           // For coordinating kernel launches
struct global_context *global_ctx = NULL; // Global shared memory
int client_id = -1;                       // This client's ID

static int client_register() {
  int id = -1;

  // Get a unique client id. This is nasty.
  acquire(&global_ctx->lock);
  if (global_ctx->nclients < NCLIENTS) {
    for (id = 0; id < NCLIENTS; id++) {
      if (global_ctx->clients[id].index == -1)
        break;
    }
  }
  if (id >= 0 && id < NCLIENTS) {
    memset(global_ctx->clients + id, 0, sizeof(global_ctx->clients[0]));
    global_ctx->clients[id].index = id;
    global_ctx->clients[id].pid = getpid();
    ILIST_ADD(global_ctx, id);
    global_ctx->nclients++;
  }
  release(&global_ctx->lock);

  return id;
}

static void client_unregister(int id) {
  if (id >= 0 && id < NCLIENTS) {
    acquire(&global_ctx->lock);
    ILIST_DEL(global_ctx, id);
    global_ctx->nclients--;
    global_ctx->clients[id].index = -1;
    release(&global_ctx->lock);
  }
}

// Registers this process in the global MQX context.
int client_init() {
  int shmfd;

  sem_launch = sem_open(MQX_SEM_LAUNCH, 0);
  if (sem_launch == SEM_FAILED) {
    mqx_print(FATAL, "Failed to open launch semaphore: %s.", strerror(errno));
    return -1;
  }
  shmfd = shm_open(MQX_SHM_GLOBAL, O_RDWR, 0);
  if (shmfd == -1) {
    mqx_print(FATAL, "Failed to open shared memory: %s.", strerror(errno));
    goto fail_shm;
  }
  global_ctx = (struct global_context *)mmap(NULL /* address */, sizeof(*global_ctx), PROT_READ | PROT_WRITE,
                                             MAP_SHARED, shmfd, 0 /* offset */);
  if (global_ctx == MAP_FAILED) {
    mqx_print(FATAL, "Failed to mmap shared memory: %s.", strerror(errno));
    goto fail_mmap;
  }
  if (msq_init() < 0) {
    mqx_print(FATAL, "Failed to initialize message queue.");
    goto fail_msq;
  }
  client_id = client_register();
  if (client_id == -1) {
    mqx_print(FATAL, "Failed to register client.");
    goto fail_client;
  }

  close(shmfd);
  return 0;

fail_client:
  msq_fini();
fail_msq:
  munmap(global_ctx, sizeof(*global_ctx));
fail_mmap:
  global_ctx = NULL;
  close(shmfd);
fail_shm:
  sem_close(sem_launch);
  sem_launch = SEM_FAILED;
  return -1;
}

void client_fini() {
  client_unregister(client_id);
  client_id = -1;
  msq_fini();
  if (global_ctx != NULL) {
    munmap(global_ctx, sizeof(*global_ctx));
    global_ctx = NULL;
  }
  if (sem_launch != SEM_FAILED) {
    sem_close(sem_launch);
    sem_launch = SEM_FAILED;
  }
}

long dev_mem_size() { return global_ctx->mem_total; }

long dev_mem_free() {
  long freesize = global_ctx->mem_total - atomic_readl(&global_ctx->mem_used);
  return freesize < 0 ? 0 : freesize;
}

long dev_mem_free2() { return global_ctx->mem_total - atomic_readl(&global_ctx->mem_used); }

void inc_attached_size(long delta) { atomic_addl(&global_ctx->mem_used, delta); }

void inc_detachable_size(long delta) { atomic_addl(&global_ctx->clients[client_id].size_detachable, delta); }

void launch_wait() {
  int ret;
  do {
    ret = sem_wait(sem_launch);
  } while (ret == -1 && errno == EINTR);
}

void launch_signal() { sem_post(sem_launch); }

// Get the id of the least recently used client with detachable
// device memory. The client is pinned if it is a remote client.
int client_lru_detachable() {
  int iclient;

  acquire(&global_ctx->lock);
  for (iclient = global_ctx->ilru; iclient != -1; iclient = global_ctx->clients[iclient].iprev) {
    if (atomic_readl(&global_ctx->clients[iclient].size_detachable) > 0)
      break;
  }
  if (iclient != -1 && iclient != client_id)
    global_ctx->clients[iclient].pinned++;
  release(&global_ctx->lock);

  return iclient;
}

void client_unpin(int client) {
  acquire(&global_ctx->lock);
  global_ctx->clients[client].pinned--;
  release(&global_ctx->lock);
}

void client_mov() {
  acquire(&global_ctx->lock);
  ILIST_MOV(global_ctx, client_id);
  release(&global_ctx->lock);
}

// Is the client a local client?
int is_client_local(int client) { return client == client_id; }

int getcid() { return client_id; }

pid_t cidtopid(int cid) { return global_ctx->clients[cid].pid; }

long compute_cost_evict(struct region *r) {
  long cost = 0;
  int i;
  if (r->flags & FLAG_PTARRAY)
    return 0;
  for (i = 0; i < r->nr_blocks; i++) {
    if (r->blocks[i].dev_valid && !r->blocks[i].swp_valid)
      cost += min(i * BLOCKSIZE + BLOCKSIZE, r->size) - i * BLOCKSIZE;
  }
  return cost;
}

void global_attach_list_add(struct region *r) {
  int index = -1;

  acquire(&global_ctx->regions.lock);
  if (global_ctx->regions.nrgns < NRGNS) {
    for (index = 0; index < NRGNS; index++) {
      if (global_ctx->regions.rgns[index].cid == -1)
        break;
    }
  }
  if (index >= 0 && index < NRGNS) {
    global_ctx->regions.rgns[index].cid = client_id;
    global_ctx->regions.rgns[index].addr = (void *)r;
    global_ctx->regions.rgns[index].size = r->size;
    atomic_set(&global_ctx->regions.rgns[index].pinned, atomic_read(&r->c_pinned));
    global_ctx->regions.rgns[index].freq = r->freq;
    global_ctx->regions.rgns[index].cost_evict = 0;
    RLIST_ADD(&global_ctx->regions, index);
    global_ctx->regions.nrgns++;
    r->index = index;
  } else {
    release(&global_ctx->regions.lock);
    panic("global attached region list full");
  }
  release(&global_ctx->regions.lock);
}

void global_attach_list_del(struct region *r) {
  acquire(&global_ctx->regions.lock);
  RLIST_DEL(&global_ctx->regions, r->index);
  global_ctx->regions.rgns[r->index].cid = -1;
  atomic_set(&global_ctx->regions.rgns[r->index].pinned, 0);
  global_ctx->regions.nrgns--;
  r->index = -1;
  release(&global_ctx->regions.lock);
}

void global_attach_list_mov(struct region *r) {
  acquire(&global_ctx->regions.lock);
  if (r->index != -1)
    RLIST_MOV(&global_ctx->regions, r->index);
  release(&global_ctx->regions.lock);
}

void global_attach_list_pin(struct region *r) { atomic_inc(&global_ctx->regions.rgns[r->index].pinned); }

void global_attach_list_unpin(struct region *r) { atomic_dec(&global_ctx->regions.rgns[r->index].pinned); }

void region_set_cost_evict(struct region *r, long cost) {
  acquire(&global_ctx->regions.lock);
  if (r->index != -1)
    global_ctx->regions.rgns[r->index].cost_evict = cost;
  release(&global_ctx->regions.lock);
}

void region_update_cost_evict(struct region *r) {
  long cost_evict = compute_cost_evict(r);
  acquire(&global_ctx->regions.lock);
  if (r->index != -1) {
    global_ctx->regions.rgns[r->index].cost_evict = cost_evict;
  }
  release(&global_ctx->regions.lock);
}

void region_inc_freq(struct region *r) {
  // No need to lock
  global_ctx->regions.rgns[r->index].freq++;
}
