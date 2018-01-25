#!/bin/bash
grep -ohE "cuda[0-9a-zA-Z_]+\(" .. -R | sort | uniq
