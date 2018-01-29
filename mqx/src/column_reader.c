#include <fcntl.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "libmps.h"

const char *formats[4] = {
  "RLE",
  "DICT",
  "DELTA",
  "UNCOMPRESSED"
};
int main(int argc, char **argv) {
  if (argc < 2) {
    printf("no column file\n");
    exit(-1);
  }
  int fd = open(argv[1], O_RDONLY);
  struct columnHeader header;
  read(fd, &header, sizeof(struct columnHeader));
  lseek(fd, 0, SEEK_SET);
  int nblock = header.blockTotal;

  char *buf = malloc(512 * 1024 * 1024);
  for (int i = 0; i < nblock; i++) {
    read(fd, &header, sizeof(struct columnHeader));
    printf("header %d: totalTupleNum(%lu) tupleNum(%lu) blockSize(%lu) blockTotal(%d) blockId(%d) format(%s)\n",
        i, header.totalTupleNum, header.tupleNum, header.blockSize, header.blockTotal, header.blockId, formats[header.format]); 
    read(fd, buf, header.blockSize);
  }
  free(buf);
}
