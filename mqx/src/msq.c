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
#include <mqueue.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h> 
#include <unistd.h>

#include "client.h"
#include "common.h"
#include "libc_hooks.h"
#include "protocol.h"

mqd_t mqid = (mqd_t) -1;
pthread_t tid_msq;
pthread_mutex_t mutx_ack;
pthread_cond_t cond_ack;
// Its meaning depends on the message; for req_evict,
// it means size of free space spared.
long ack = 0;

int msq_send(int client, const struct msg *msg)
{
    char qname[32];
    mqd_t qid;

    sprintf(qname, "/mqx_cli_%d", cidtopid(client));
    qid = mq_open(qname, O_WRONLY);
    if (qid == (mqd_t) -1) {
        mqx_print(FATAL, "failed to open message queue in process %d: %s", \
                cidtopid(client), strerror(errno));
        return -1;
    }

    if (mq_send(qid, (const char *)msg, msg->size, 0) == -1) {
        mqx_print(FATAL, "failed to send a message: %s", strerror(errno));
        mq_close(qid);
        return -1;
    }

    mq_close(qid);
    return 0;
}

// Send a MSQ_REQ_EVICT message to the remote %client. %block specifies
// whether we should block for the reply (MSG_REP_ACK).
long msq_send_req_evict(int client, void *addr, long size_needed, int block)
{
    struct msg_req msg;
    long size_spared = 0;
    int ret = 0;

    msg.type = MSG_REQ_EVICT;
    msg.size = sizeof(msg);
    msg.from = getcid();
    msg.addr = addr;
    msg.size_needed = size_needed;
    msg.block = block;

    if (block) {
        pthread_mutex_lock(&mutx_ack);
    }
    ret = msq_send(client, (struct msg *)&msg);
    if (block) {
        if (ret == 0) {
            pthread_cond_wait(&cond_ack, &mutx_ack);
            size_spared = ack;
        }
        pthread_mutex_unlock(&mutx_ack);
    }

    return (ret == 0) ? size_spared : (-1);
}

int msq_send_rep_ack(int client, long ack)
{
    struct msg_rep msg;

    msg.type = MSG_REP_ACK;
    msg.size = sizeof(msg);
    msg.from = getcid();
    msg.ret = ack;

    return msq_send(client, (struct msg *)&msg);
}

int local_victim_evict(void *addr, long size_needed);

void handle_req_evict(struct msg_req *msg)
{
    long ret;

    if (msg->size != sizeof(*msg)) {
        mqx_print(FATAL, "message size unmatches size of msg_req");
        return;
    }
    if (msg->from == getcid()) {
        mqx_print(FATAL, "message from self");
        return;
    }

    ret = local_victim_evict(msg->addr, msg->size_needed);
    if (msg->block)
        msq_send_rep_ack(msg->from, ret);
}

void handle_rep_ack(struct msg_rep *msg)
{
    if (msg->size != sizeof(*msg)) {
        mqx_print(FATAL, "message size unmatches size of msg_rep");
        return;
    }
    if (msg->from == getcid()) {
        mqx_print(FATAL, "message from self");
        return;
    }

    pthread_mutex_lock(&mutx_ack);
    ack = msg->ret;
    pthread_cond_signal(&cond_ack);
    pthread_mutex_unlock(&mutx_ack);
}

// The thread that receives and handles messages from peer clients.
void *thread_msq_listener(void *arg)
{
    struct mq_attr qattr;
    char *msgbuf = NULL;
    ssize_t msgsz;

    deactivate_libc_hooks();

    if (mq_getattr(mqid, &qattr) == -1) {
        mqx_print(FATAL, "Failed to get msq attr: %s.", strerror(errno));
        pthread_exit(NULL);
    }
    msgbuf  = (char *)__libc_malloc(qattr.mq_msgsize + 1);
    if (!msgbuf) {
        mqx_print(FATAL, "Failed to allocate msgbuf: %s.", strerror(errno));
        pthread_exit(NULL);
    }

    while (1) {
        // Receive a message.
        msgsz = mq_receive(mqid, msgbuf, qattr.mq_msgsize + 1, NULL);
        if (msgsz == -1) {
            if (errno == EINTR)
                continue;
            else if (errno == EBADF) {
                mqx_print(INFO, "Message queue closed.");
                break;
            }
            else {
                mqx_print(ERROR, "Error in receiving message: %s.",
                    strerror(errno));
                break;
            }
        }
        else if (msgsz != ((struct msg *)msgbuf)->size) {
            mqx_print(INFO,
                "Bytes received (%ld) do not match message size (%d)",
                msgsz, ((struct msg *)msgbuf)->size);
            continue;
        }
        // Handle the message.
        switch (((struct msg *)msgbuf)->type) {
        case MSG_REQ_EVICT:
            handle_req_evict((struct msg_req *)msgbuf);
            break;
        case MSG_REP_ACK:
            handle_rep_ack((struct msg_rep *)msgbuf);
            break;
        default:
            mqx_print(ERROR, "Unknown message type (%d)",
                ((struct msg *)msgbuf)->type);
            break;
        }
    }

    __libc_free(msgbuf);
    mqx_print(INFO, "Message listener exiting.");
    return NULL;
}

int msq_init()
{
    char qname[32];

    if (mqid != (mqd_t)-1) {
        mqx_print(ERROR, "Message queue already initialized.");
        return -1;
    }
    sprintf(qname, "/mqx_cli_%d", getpid());
    mqid = mq_open(qname, O_RDONLY | O_CREAT | O_EXCL, 0622, NULL);
    if (mqid == (mqd_t) -1) {
        mqx_print(FATAL, "Failed to create message queue: %s.",
            strerror(errno));
        return -1;
    }

    pthread_mutex_init(&mutx_ack, NULL);
    pthread_cond_init(&cond_ack, NULL);
    if (pthread_create(&tid_msq, NULL, thread_msq_listener, NULL) != 0) {
        mqx_print(FATAL, "Failed to create msq listener thread: %s.",
            strerror(errno));
        pthread_cond_destroy(&cond_ack);
        pthread_mutex_destroy(&mutx_ack);
        mq_close(mqid);
        mq_unlink(qname);
        mqid = (mqd_t)-1;
        return -1;
    }

    return 0;
}

void msq_fini()
{
    if (mqid != (mqd_t) -1) {
        char qname[32];
        sprintf(qname, "/mqx_cli_%d", getpid());
        mq_close(mqid);
        mq_unlink(qname);
        mqid = (mqd_t) -1;
        pthread_cancel(tid_msq);
        pthread_join(tid_msq, NULL);
        pthread_cond_destroy(&cond_ack);
        pthread_mutex_destroy(&mutx_ack);
    }
}
