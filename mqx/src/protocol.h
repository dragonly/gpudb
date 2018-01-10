/*
 * Copyright (c) 2014 Kaibo Wang (wkbjerry@gmail.com)
 * Copyright (c) 2017 Yilong Li <liyilongko@gmail.com>
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
// The protocol between MQX clients and MQX server.

#ifndef _MQX_PROTOCOL_H_
#define _MQX_PROTOCOL_H_

#include "atomic.h"
#include "list.h"
#include "spinlock.h"
#include <unistd.h>
#include <stdint.h>
#include <cuda.h>

/**
 * MPS related
 */
#define SERVER_SOCKET_FILE "mqx_mps_server"
#define MAX_BUFFER_SIZE 8192

#define MAX_ARG_SIZE  4096
#define MAX_ARG_NUM   16
#define MAX_CLIENTS   32

struct mps_client {
  uint16_t id;
  CUstream stream;
};
struct server_stats {
  struct mps_client clients[MAX_CLIENTS];
  // hard coded as 32 bit unsigned int for simplicity
  uint32_t clients_bitmap;
  uint16_t nclients;
};
struct kernel_args {
  void *arg_info[MAX_ARG_NUM + 1];
  uint16_t last_arg_len;
  char args[MAX_ARG_SIZE];
  uint16_t blocks_per_grid;
  uint16_t threads_per_block;
  uint16_t function_index;
};

#define REQ_HOST_MALLOC             0
#define REQ_GPU_LAUNCH_KERNEL       1
#define REQ_GPU_MALLOC              2
#define REQ_GPU_MEMCPY_HTOD_SYNC    3
#define REQ_GPU_MEMCPY_HTOD_ASYNC   4
#define REQ_GPU_MEMCPY_DTOH_SYNC    5
#define REQ_GPU_MEMCPY_DTOH_ASYNC   6
#define REQ_GPU_SYNC                7
#define REQ_GPU_MEMFREE             8
#define REQ_GPU_MEMSET              9
#define REQ_QUIT                    10
#define MPS_REQ_SIZE 4
struct mps_req {
  uint16_t type;
  uint16_t len;
};

#define RES_OK   0
#define RES_FAIL 1
#define MPS_RES_SIZE 2
struct mps_res {
  uint16_t type;
};

/**
 * original mqx related
 */
struct region_elem {
  int cid;
  int iprev, inext;
  atomic_t pinned;
  void *addr;
  long size;
  int freq;        // access frequency
  long cost_evict; // cost of data eviction
};

#define NRGNS 128

struct region_list {
  struct spinlock lock;
  struct region_elem rgns[NRGNS];
  int nrgns, imru, ilru;
};

// The maximum number of concurrent processes managed by MQX
#define NCLIENTS 32

// A MQX client registered in the global shared memory.
// Each client has a POXIS message queue, named "/mqx_cli_%pid",
// that receives requests and/or notifications from other peer clients.
struct mqx_client {
  int index; // index of this client; -1 means unoccupied
  int iprev; // index of the previous client in the LRU list
  int inext; // index of the next client in the LRU list
  int pinned;
  latomic_t size_detachable;
  pid_t pid;
};

// The global context shared by all MQX clients
struct global_context {
  // Total size of device memory.
  long mem_total;
  // Size of used (attached) device memory
  // NOTE: in numbers, device memory may be
  // over-used, i.e., mem_used > mem_total.
  latomic_t mem_used;
  struct spinlock lock;
  struct mqx_client clients[NCLIENTS];
  int nclients;
  int imru;
  int ilru;
  // Global list of attached regions
  struct region_list regions;
  struct mps_client mps_clients[NCLIENTS];
  uint32_t mps_clients_bitmap;
  uint16_t mps_nclients;
  pthread_mutex_t mps_lock;
};

enum msgtype {
  MSG_REQ_EVICT,
  MSG_REP_ACK,
};

// Message header
struct msg {
  int type;
  int size;
};

// A message requesting for eviction.
struct msg_req {
  int type;
  int size;
  int from;
  void *addr;
  long size_needed;
  int block;
};

// A message replying an eviction request.
struct msg_rep {
  int type;
  int size;
  int from;
  long ret;
};

#define MQX_SEM_LAUNCH "/mqx_sem_launch"
#define MQX_SHM_GLOBAL "/mqx_shm_global"

// Add the inew'th client to the MRU end of p's client list
static inline void ILIST_ADD(struct global_context *p, int inew) {
  if (p->imru == -1) {
    p->ilru = p->imru = inew;
    p->clients[inew].iprev = -1;
    p->clients[inew].inext = -1;
  } else {
    p->clients[inew].iprev = -1;
    p->clients[inew].inext = p->imru;
    p->clients[p->imru].iprev = inew;
    p->imru = inew;
  }
}

// Delete a client from p's client list
static inline void ILIST_DEL(struct global_context *p, int idel) {
  int iprev = p->clients[idel].iprev;
  int inext = p->clients[idel].inext;

  if (iprev != -1)
    p->clients[iprev].inext = inext;
  else
    p->imru = inext;

  if (inext != -1)
    p->clients[inext].iprev = iprev;
  else
    p->ilru = iprev;
}

// Move a client to the MRU end of p's client list
static inline void ILIST_MOV(struct global_context *p, int imov) {
  ILIST_DEL(p, imov);
  ILIST_ADD(p, imov);
}

// Add the inew'th client to the MRU end of p's client list
static inline void RLIST_ADD(struct region_list *p, int inew) {
  if (p->imru == -1) {
    p->ilru = p->imru = inew;
    p->rgns[inew].iprev = -1;
    p->rgns[inew].inext = -1;
  } else {
    p->rgns[inew].iprev = -1;
    p->rgns[inew].inext = p->imru;
    p->rgns[p->imru].iprev = inew;
    p->imru = inew;
  }
}

// Delete a client from p's client list
static inline void RLIST_DEL(struct region_list *p, int idel) {
  int iprev = p->rgns[idel].iprev;
  int inext = p->rgns[idel].inext;

  if (iprev != -1)
    p->rgns[iprev].inext = inext;
  else
    p->imru = inext;

  if (inext != -1)
    p->rgns[inext].iprev = iprev;
  else
    p->ilru = iprev;
}

// Move a client to the MRU end of p's client list
static inline void RLIST_MOV(struct region_list *p, int imov) {
  RLIST_DEL(p, imov);
  RLIST_ADD(p, imov);
}

#endif

