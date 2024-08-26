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

#define PAGE_SIZE 4096  * 256
static unsigned char page[PAGE_SIZE] __attribute__ ((aligned (4096)));

static FILE* pMemFile = NULL;
static FILE* pMapsFile = NULL;

static bool read_range(struct range* address) {
    char line[256];
    if (fgets(line, 256, pMapsFile) == NULL) {
        return false;
    }
    sscanf(line, "%lx-%lx %*[^\n]\n", &address->start, &address->end);
    return true;
}

static void dump_memfile(long length) {
    if (length == 0) {
        return;
    }
    if (length < 0) {
        fprintf(stderr, "negative length: %ld\n", length);
        exit(1);
    }
    long bytes_read = (long)fread(&page, 1, PAGE_SIZE, pMemFile);
    if (bytes_read == 0) {
        return;
    }
    if (bytes_read < 0) {
        fprintf(stderr, "bytes_read too large: %lu\n", (size_t)bytes_read);
        exit(1);
    }
    fwrite(&page, 1, bytes_read, stdout);
    return dump_memfile(length - bytes_read);
}

static FILE* open_proc(int pid, const char* basename) {
    static char buf[256];
    sprintf(buf, "/proc/%d/%s", pid, basename);
    FILE* res = fopen(buf, "r");
    if (!res) {
        fprintf(stderr, "Failed to open proc files\n");
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
        printf("Unable to attach to the pid specified\n");
        exit(1);
    }
    wait(NULL);

    pMapsFile = open_proc(pid, "maps");
    pMemFile = open_proc(pid, "mem");

    struct range current;
    if (!read_range(&current)) {
        fprintf(stderr, "empty maps file\n");
        exit(1);
    }

    for (struct range next; read_range(&next); ) {
        if (next.start == current.end) {
            current.end = next.end;
            continue;
        }
        fprintf(stderr, "dump %lx-%lx\n", current.start, current.end);
        fseeko(pMemFile, current.start, SEEK_SET);
        dump_memfile(current.end - current.start);
        current = next;
    }
    fseeko(pMemFile, current.start, SEEK_SET);
    dump_memfile(current.end - current.start);

    fclose(pMapsFile);
    fclose(pMemFile);

    ptrace(PTRACE_CONT, pid, NULL, NULL);
    ptrace(PTRACE_DETACH, pid, NULL, NULL);
    return 0;
}
