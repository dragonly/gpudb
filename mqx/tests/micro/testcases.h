#include "test.h"

// Test cases
int test_evict_local();

struct test_case testcases[] = {
	{
		test_evict_local,
		"local evictions"
	},
};
