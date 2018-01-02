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
// Controls the setup and destroying of MQX global shared memory area.

#include <sys/mman.h>
#include <sys/stat.h> /* For mode constants */
#include <fcntl.h> /* For O_* constants */
#include <getopt.h>
#include <semaphore.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "common.h"
#include "protocol.h"

int verbose = 0;

int start(size_t mem_usable)
{
    size_t size_free, size_total, size_usable;
    struct global_context *pglobal;
    int user_specified = 0;
    cudaError_t ret;
    int i, shmfd;
    sem_t *sem;

    // Get the maximum amount of device memory space usable to MQX.
    ret = cudaMemGetInfo(&size_free, &size_total);
    if (ret != cudaSuccess) {
        mqx_print(FATAL, "Failed to get CUDA device memory info: %s",
            cudaGetErrorString(ret));
        exit(-1);
    }
    size_usable = size_free;
    if (mem_usable > 0 && mem_usable <= size_total) {
    	size_usable = mem_usable;
    	user_specified = 1;
    }
    if (verbose) {
        mqx_print(INFO, "Total device memory  : %lu bytes", size_total);
        mqx_print(INFO, "Free device memory   : %lu bytes", size_free);
        mqx_print(INFO, "Usable device memory : %lu bytes%s", size_usable,
            user_specified ? " (user specified)" : "");
        mqx_print(INFO, "Setting up MQX context ......");
    }

    // Create the synchronization semaphore for launching kernels.
    sem = sem_open(MQX_SEM_LAUNCH, O_CREAT | O_EXCL, 0660, 1);
    if (sem == SEM_FAILED && errno == EEXIST) {
        // Semaphore already exists; have to re-link it.
        mqx_print(WARN, "Semaphore already exists; re-linking.");
        if (sem_unlink(MQX_SEM_LAUNCH) == -1) {
            mqx_print(FATAL, "Failed to remove semaphore; quitting");
            exit(1);
        }
        sem = sem_open(MQX_SEM_LAUNCH, O_CREAT | O_EXCL, 0660, 1);
        if (sem == SEM_FAILED) {
            mqx_print(FATAL, "Failed to create semaphore; quitting.");
            exit(1);
        }
    }
    else if (sem == SEM_FAILED) {
        mqx_print(FATAL, "Failed to create semaphore; quitting.");
        exit(1);
    }
    // The semaphore link has been created; we can close it now.
    sem_close(sem);

    // Create and initialize shared memory.
    // TODO: a better procedure is to disable access to shared memory first
    // (by setting permission), set up content, and then enable access.
    shmfd = shm_open(MQX_SHM_GLOBAL, O_RDWR | O_CREAT | O_EXCL, 0644);
    if (shmfd == -1 && errno == EEXIST) {
        mqx_print(WARN, "Shared memory already exists; re-linking.");
        if (shm_unlink(MQX_SHM_GLOBAL) == -1) {
            mqx_print(FATAL, "Failed to remove shared memory; quitting.");
            goto fail_shm;
        }
        shmfd = shm_open(MQX_SHM_GLOBAL, O_RDWR | O_CREAT | O_EXCL, 0644);
        if (shmfd == -1) {
            mqx_print(FATAL, "Failed to create shared memory; quitting.");
            goto fail_shm;
        }
    }
    else if (shmfd == -1) {
        mqx_print(FATAL, "Failed to create shared memory; quitting.");
        goto fail_shm;
    }
    // Truncating a shared memory initializes its content to all zeros.
    if (ftruncate(shmfd, sizeof(struct global_context)) == -1) {
        mqx_print(FATAL, "Failed to truncate the shared memory; quitting.");
        goto fail_truncate;
    }
    pglobal = (struct global_context *)mmap(
        NULL /* address */, sizeof(*pglobal), PROT_READ | PROT_WRITE,
        MAP_SHARED, shmfd, 0 /* offset */);
    if (pglobal == MAP_FAILED) {
        mqx_print(FATAL, "Failed to map the shared memory; quitting.");
        goto fail_mmap;
    }
        mqx_print(WARN, "pglobal(%p) size(%ld)", pglobal, sizeof(*pglobal));
    // Further initialize the content of the shared memory.
    pglobal->mem_total = size_usable;
    atomic_setl(&pglobal->mem_used, 0);
    initlock(&pglobal->lock);
    pglobal->nclients = 0;
    pglobal->ilru = -1;
    pglobal->imru = -1;
    memset(pglobal->clients, 0, sizeof(pglobal->clients[0]) * NCLIENTS);
    for (i = 0; i < NCLIENTS; i++) {
        pglobal->clients[i].index = -1;
        pglobal->clients[i].inext = -1;
        pglobal->clients[i].iprev = -1;
    }
    memset(&pglobal->regions, 0, sizeof(pglobal->regions));
    pglobal->regions.ilru = pglobal->regions.imru = -1;
    for (i = 0; i < NRGNS; i++) {
        pglobal->regions.rgns[i].cid = -1;
        pglobal->regions.rgns[i].iprev = -1;
        pglobal->regions.rgns[i].inext = -1;
    }

    // Initialization finished.
    munmap(pglobal, sizeof(*pglobal));
    close(shmfd);
    if (verbose)
        mqx_print(INFO, "Setting done!\nMQX context initialized.");
    return 0;

fail_mmap:
fail_truncate:
    close(shmfd);
    shm_unlink(MQX_SHM_GLOBAL);
fail_shm:
    sem_unlink(MQX_SEM_LAUNCH);
    return -1;
}

int stop()
{
    if (shm_unlink(MQX_SHM_GLOBAL) == -1) {
        mqx_print(ERROR, "Failed to unlink MQX shared memory: %s.",
                strerror(errno));
    }
    if (sem_unlink(MQX_SEM_LAUNCH) == -1) {
        mqx_print(ERROR, "Failed to unlink semaphore: %s.", strerror(errno));
    }
    if (verbose)
        mqx_print(INFO, "MQX context stopped.");
    return 0;
}

int restart(size_t mem_usable)
{
    stop();
    return start(mem_usable);
}

// Print brief status of registered MQX clients.
int info()
{
    struct global_context *pglobal;
    int shmfd, iclient;

    shmfd = shm_open(MQX_SHM_GLOBAL, O_RDWR, 0);
    if (shmfd == -1) {
        mqx_print(FATAL, "MQX not started: %s", strerror(errno));
        return -1;
    }
    pglobal = (struct global_context *)mmap(
        NULL /* address */, sizeof(*pglobal), PROT_READ | PROT_WRITE,
        MAP_SHARED, shmfd, 0 /* offset */);
    if (pglobal == MAP_FAILED) {
        mqx_print(FATAL, "Cannot map MQX shared memory: %s", strerror(errno));
        close(shmfd);
        return -1;
    }
    close(shmfd);

    mqx_print(INFO, "MQX status:");
    mqx_print(INFO, "    mem_total:      %ld", pglobal->mem_total);
    mqx_print(INFO, "    mem_used:       %ld", pglobal->mem_used);
    mqx_print(INFO, "    Clients in MRU order:");
    acquire(&pglobal->lock);
    if (pglobal->imru == -1)
        mqx_print(INFO, "    [None]");
    for (iclient = pglobal->imru; iclient != -1;
            iclient = pglobal->clients[iclient].inext) {
        struct mqx_client *client = &pglobal->clients[iclient];
        mqx_print(INFO, "    pid: %d", client->pid);
        mqx_print(INFO, "        pinned:        %d", client->pinned);
        mqx_print(INFO, "        detachable:    %ld", client->size_detachable);
    }
    release(&pglobal->lock);
    return 0;
}

int main(int argc, char *argv[])
{
    struct option options[7];
    char *opts = "serm:vi";
    int command = '\0';
    long mem_usable = 0;
    int c, ret = 0;

    memset(options, 0, sizeof(options[0]) * 7);
    options[0].name = "start";
    options[0].val = 's';
    options[1].name = "stop";
    options[1].val = 'e';
    options[2].name = "restart";
    options[2].val = 'r';
    options[3].name = "mem";
    options[3].val = 'm';
    options[3].has_arg = 1;
    options[4].name = "verbose";
    options[4].val = 'v';
    options[4].name = "info";
    options[4].val = 'i';

    while ((c = getopt_long(argc, argv, opts, options, NULL)) != -1) {
        switch (c) {
        case 's':
        case 'e':
        case 'r':
        case 'i':
            command = c;
            break;
        case 'm':
            mem_usable = atol(optarg);
            if (mem_usable < 0) {
                mqx_print(ERROR, "Device memory size cannot be negative.");
            	mem_usable = 0;
            }
            break;
        case 'v':
            verbose = 1;
            break;
        case '?':
            if (optopt == 'm')
                mqx_print(FATAL, "Please specify device memory size");
            else
                mqx_print(FATAL, "Unknown option %c", c);
            return -1;
            break;
        default:
            abort();
            break;
        }
    }

    if (command == 's')
        ret = start(mem_usable);
    else if (command == 'e')
        ret = stop();
    else if (command == 'r')
        ret =restart(mem_usable);
    else if (command == 'i')
        ret = info();
    else {
        mqx_print(ERROR, "No operation specified");
        ret = -1;
    }
    return ret;
}
