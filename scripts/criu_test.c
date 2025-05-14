/*
 * Minimal CRIU incremental checkpoint test using the C API
 * Requires: CRIU installed and in PATH
 * Compile with: gcc -O0 -o criu_test criu_test.c -I/usr/local/include -L/usr/local/lib -lcriu
 * Run with: ./criu_test
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <errno.h>
#include <limits.h>
#include <criu/criu.h>

#define DUMP_DIR "dump"
#define PAGE_SIZE 4096
#define ITERATIONS 100

void dirty_memory_static() {
    static volatile char buf[PAGE_SIZE];
    static int val = 0;
    val++;
    memset((void*)buf, val, PAGE_SIZE);
}

void do_criu_setup(int pid, bool dedup) {
    criu_set_pid(pid);
    criu_set_log_file("criu.log");
    criu_set_log_level(4);
    criu_set_shell_job(true);
    criu_set_leave_running(true);
    criu_set_service_address("/tmp/criu_service.socket");
    criu_set_track_mem(true);
    criu_set_auto_dedup(dedup);
}

int do_criu_dump(int iteration, int pid) {
    char dir[128];
    snprintf(dir, sizeof(dir), "%s/%d", DUMP_DIR, iteration);
    mkdir(DUMP_DIR, 0755);
    mkdir(dir, 0755);

    int dir_fd = open(dir, O_DIRECTORY);
    if (dir_fd < 0) {
        perror("open images dir");
        return -1;
    }

    criu_set_images_dir_fd(dir_fd);

    if (iteration > 0) {
        char parent_dir[PATH_MAX];
        snprintf(parent_dir, sizeof(parent_dir), "%s/%d", DUMP_DIR, iteration - 1);
        if (access(parent_dir, F_OK) != 0) {
            fprintf(stderr, "Parent dump dir %s not found\n", parent_dir);
            close(dir_fd);
            return -1;
        }
        char parent_path[32];
        snprintf(parent_path, sizeof(parent_path), "../%d", iteration - 1);
        criu_set_parent_images(parent_path);
    }

    int ret = criu_dump();
    close(dir_fd);
    if (ret < 0) {
        perror("criu_dump");
        fprintf(stderr, "CRIU dump failed at iteration %d. error %d\n", iteration, ret);
        return -1;
    }

    return 0;
}

int main(int argc, char *argv[]) {
    bool enable_auto_dedup = true; // Default to true (original problematic behavior)

    if (argc > 1) {
        if (strcmp(argv[1], "no-dedup") == 0) {
            enable_auto_dedup = false;
            printf("auto_dedup will be disabled for this run.\n");
        } else {
            fprintf(stderr, "Unknown argument '%s'. Running with default settings (auto_dedup enabled).\n", argv[1]);
        }
    } else {
        printf("Running with default settings (auto_dedup enabled).\n");
    }
    
    if (criu_init_opts() < 0) {
        perror("criu_init_opts");
        return 1;
    }

    int pid = getpid();

    do_criu_setup(pid, enable_auto_dedup);

    for (int i = 0; i < ITERATIONS; ++i) {
	printf("\r%d", i);
	fflush(stdout);
	
        dirty_memory_static();
        do_criu_dump(i, pid);
            
        usleep(10000);
    }
    printf("\n");
    return 0;
}