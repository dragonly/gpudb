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
#ifndef _MQX_CLIENT_H_
#define _MQX_CLIENT_H_

#include <unistd.h>

#include "core.h"
#include "protocol.h"

// Functions exposed to client-side code to interact with global shared memory.
int client_init();
void client_fini();

void launch_wait();
void launch_signal();

long dev_mem_size();
long dev_mem_free();
long dev_mem_free2();

void inc_attached_size(long delta);
void inc_detachable_size(long delta);

int client_lru_detachable();
void client_unpin(int client);
void client_mov();

void global_attach_list_add(struct region *r);
void global_attach_list_del(struct region *r);
void global_attach_list_mov(struct region *r);
void global_attach_list_pin(struct region *r);
void global_attach_list_unpin(struct region *r);

void region_set_cost_evict(struct region *r, long cost);
void region_update_cost_evict(struct region *r);
void region_inc_freq(struct region *r);

int is_client_local(int client);
int getcid();
pid_t cidtopid(int cid);

#endif
