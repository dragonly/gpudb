#ifndef _MQX_TEST_H_
#define _MQX_TEST_H_

#include <stdio.h>

struct test_case
{
    int (*func)(void);
    char *comment;
};

#define MQX_TPRINT(fmt, arg...) fprintf(stderr, "T " fmt "\n", ##arg)

#endif
