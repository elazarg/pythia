#include <stdio.h>
#include <elf.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define MAX_MAPPINGS 1024

struct dump {
    char *base;
    Elf64_Shdr *mappings[MAX_MAPPINGS];
};

static unsigned readdump(const char *path, struct dump *dump) {
    unsigned count = 0;
    int fd = open(path, O_RDONLY);
    if (fd != -1) {
        struct stat stat;
        fstat(fd, &stat);
        dump->base = mmap(NULL, stat.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        Elf64_Ehdr *header = (Elf64_Ehdr *)dump->base;
        Elf64_Shdr *secs = (Elf64_Shdr*)(dump->base+header->e_shoff);
        for (unsigned secinx = 0; secinx < header->e_shnum; secinx++) {
            if (secs[secinx].sh_type == SHT_PROGBITS) {
                if (count == MAX_MAPPINGS) {
                    count = 0;
                    break;
                }
                dump->mappings[count] = &secs[secinx];
                count++;
            }
        }
        dump->mappings[count] = NULL;
    }
    return count;
}

#define chunk_size 64

static unsigned long printdiff(struct dump *dump1, Elf64_Shdr *sec1, struct dump *dump2, Elf64_Shdr *sec2) {
    unsigned char *data1 = (unsigned char *)(dump1->base+sec1->sh_offset);
    unsigned char *data2 = (unsigned char *)(dump2->base+sec2->sh_offset);
    unsigned difffound = 0;
    long count = 0;
    unsigned start = 0;
    for (unsigned i = 0; i < sec1->sh_size; i++) {
        if (i%chunk_size == 0) {
            start = i;
            difffound = 0;
        }
        if (!difffound && data1[i] != data2[i]) {
            difffound = 1;
        }
        if ((i+1)%chunk_size == 0 || i + 1 == sec1->sh_size) {
            if (difffound) {
                count += chunk_size;
            }
        }
    }
    return count;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: compare DUMP1 DUMP2\n");
        return 1;
    }
    struct dump dump1;
    struct dump dump2;
    if (readdump(argv[1], &dump1) == 0 ||
            readdump(argv[2], &dump2) == 0) {
        fprintf(stderr, "Failed to read dumps\n");
        return 1;
    }
    unsigned sinx1 = 0;
    unsigned sinx2 = 0;
    long count = 0;
    while (dump1.mappings[sinx1] || dump2.mappings[sinx2]) {
        Elf64_Shdr *sec1 = dump1.mappings[sinx1];
        Elf64_Shdr *sec2 = dump2.mappings[sinx2];
        if (sec1 && sec2) {
            if (sec1->sh_addr == sec2->sh_addr) {
                // in both
                count += printdiff(&dump1, sec1, &dump2, sec2);
                sinx1++;
                sinx2++;
            } else if (sec1->sh_addr < sec2->sh_addr) {
                // in 1, not 2
                count += sec1->sh_size;
                sinx1++;
            } else {
                // in 2, not 1
                count += sec2->sh_size;
                sinx2++;
            }
        } else if (sec1) {
            // in 1, not 2
            count += sec1->sh_size;
            sinx1++;
        } else {
            // in 2, not 1
            count += sec2->sh_size;
            sinx2++;
        }
    }
    printf("Diff: %ld\n", count);
    return 0;
}
