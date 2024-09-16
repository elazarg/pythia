#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <stdbool.h>

struct range {
    unsigned long start;
    unsigned long end;
};

#define PAGE_SIZE (4096 * 256)
static unsigned char page[PAGE_SIZE] __attribute__ ((aligned (4096)));

static FILE* pMemFile = NULL;
static FILE* pMapsFile = NULL;

static bool read_range(struct range* address) {
    char line[257];
    if (fgets(line, 256, pMapsFile) == NULL) {
        return false;
    }
    line[256] = '\0';
    int n_read = sscanf(line, "%lx-%lx %*[^\n]\n", &address->start, &address->end);
    if (n_read != 2) {
        fprintf(stderr, "Failed to parse: %s\n", line);
        exit(1);
    }
    return true;
}

static void dump_size(long length) {
    if (length == 0) {
        return;
    }
    if (length < 0) {
        fprintf(stderr, "negative length: %ld\n", length);
        exit(1);
    }
    const long bytes_read = (long)fread(&page, 1, PAGE_SIZE, pMemFile);
    const size_t bytes_written = fwrite(&page, 1, bytes_read, stdout);
    if (bytes_written < bytes_read) {
        fprintf(stderr, "Write failed\n");
        exit(1);
    }
    if (bytes_read < PAGE_SIZE) {
        return;
    }
    return dump_size(length - PAGE_SIZE);
}

static void dump_mem(struct range address) {
    fseeko(pMemFile, address.start, SEEK_SET);
    dump_size(address.end - address.start);
}

static FILE* open_proc(int pid, const char* basename) {
    static char buf[257];
    sprintf(buf, "/proc/%d/%s", pid, basename);
    buf[256] = 0;
    FILE* res = fopen(buf, "r");
    if (!res) {
        fprintf(stderr, "Failed to open proc file %s\n", buf);
        exit(1);
    }
    return res;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "%s <pid>\n", argv[0]);
        exit(1);
    }
    const int pid = atoi(argv[1]);
    if (ptrace(PTRACE_ATTACH, pid, NULL, NULL) < 0) {
        printf("Unable to attach to pid %d\n", pid);
        exit(1);
    }
    wait(NULL);

    pMapsFile = open_proc(pid, "maps");
    pMemFile = open_proc(pid, "mem");

    struct range current;
    if (!read_range(&current)) {
        fprintf(stderr, "Empty maps file\n");
        exit(1);
    }

    for (struct range next; read_range(&next); current.end = next.end) {
        if (next.start != current.end) {
            dump_mem(current);
            current.start = next.start;
        }
    }
    dump_mem(current);

    fclose(pMapsFile);
    fclose(pMemFile);

    ptrace(PTRACE_CONT, pid, NULL, NULL);
    ptrace(PTRACE_DETACH, pid, NULL, NULL);
    return 0;
}
