#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <sys/ptrace.h>
#include <sys/wait.h>

#define pageLength 4096

static void dump_memory_region(FILE* pMemFile, unsigned long start_address, long length) {
    static unsigned char page[pageLength];
    fseeko(pMemFile, start_address, SEEK_SET);
    size_t bytes_read=1;
    for (unsigned long address=start_address; address < start_address + length; address += bytes_read) {
        bytes_read = fread(&page, 1, pageLength, pMemFile);
        fwrite(&page, 1, bytes_read, stdout);
        if (bytes_read == 0) {
            break;
        }
    }
}

static FILE* open_proc(int pid, const char* basename) {
    static char buf[256];
    sprintf(buf, "/proc/%d/%s", pid, basename);
    return fopen(buf, "r");
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

    FILE* pMapsFile = open_proc(pid, "maps");
    FILE* pMemFile = open_proc(pid, "mem");

    char line[256];
    while (fgets(line, 256, pMapsFile) != NULL) {
        unsigned long start_address;
        unsigned long end_address;
        sscanf(line, "%lx-%lx %*[^\n]\n", &start_address, &end_address);
        fprintf(stderr, "%lx-%lx\n", start_address, end_address);
        dump_memory_region(pMemFile, start_address, end_address - start_address);
    }
    fclose(pMapsFile);
    fclose(pMemFile);

    ptrace(PTRACE_CONT, pid, NULL, NULL);
    ptrace(PTRACE_DETACH, pid, NULL, NULL);
    return 0;
}
