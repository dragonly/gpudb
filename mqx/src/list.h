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
#ifndef _MQX_LIST_H_
#define _MQX_LIST_H_

struct list_head {
    struct list_head *prev;
    struct list_head *next;
};

#define LIST_HEAD_INIT(name) { &(name), &(name) }

static inline void INIT_LIST_HEAD(struct list_head *list)
{
    list->next = list;
    list->prev = list;
}

// Insert a new entry between two known consecutive entries.
static inline void __list_add(
        struct list_head *add,
        struct list_head *prev,
        struct list_head *next)
{
    next->prev = add;
    add->next = next;
    add->prev = prev;
    prev->next = add;
}

// Add a new entry.
static inline void list_add(struct list_head *add, struct list_head *head)
{
    __list_add(add, head, head->next);
}

// Add a new entry to the tail.
static inline void list_add_tail(struct list_head *add, struct list_head *head)
{
    __list_add(add, head->prev, head);
}

// Delete a list entry by making the prev/next entries point to each other.
static inline void __list_del(struct list_head * prev, struct list_head * next)
{
    next->prev = prev;
    prev->next = next;
}

// Delete an entry from list.
static inline void list_del(struct list_head *entry)
{
    __list_del(entry->prev, entry->next);
}

// Delete from one list and add as another's head.
static inline void list_move(struct list_head *list, struct list_head *head)
{
    list_del(list);
    list_add(list, head);
}

// Delete from one list and add as another's tail.
static inline void list_move_tail(
        struct list_head *list,
        struct list_head *head)
{
    list_del(list);
    list_add_tail(list, head);
}

// Iterate over a list.
#define list_for_each(pos, head) \
    for (pos = (head)->next; pos != (head); pos = pos->next)

// Iterate over a list backwards.
#define list_for_each_prev(pos, head) \
    for (pos = (head)->prev; pos != (head); pos = pos->prev)

// Test whether a list is empty.
static inline int list_empty(const struct list_head *head)
{
    return head->next == head;
}

#define OFFSETOF(TYPE, MEMBER) ((unsigned long) &((TYPE *)0)->MEMBER)

// Get the struct containing a list_head entry.
#define list_entry(ptr, type, member) \
    ((type *) ((char *)(ptr) - OFFSETOF(type,member)))

#endif
