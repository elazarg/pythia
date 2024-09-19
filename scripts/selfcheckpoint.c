#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>     // For access(), getpid(), sleep(), etc.
#include <fcntl.h>      // For open()
#include <sys/types.h>
#include <criu/criu.h>

#define CHECKPOINT_ITER 5
#define TOTAL_ITER 10

int fd = -1;

void set_criu() {
    if (criu_init_opts() < 0) {
        perror("CRIU init failed");
        exit(EXIT_FAILURE);
    }

    fd = open("./criu_images", O_DIRECTORY);
    if (fd < 0) {
        perror("Failed to open criu_images directory");
        exit(EXIT_FAILURE);
    }
    criu_set_images_dir_fd(fd);  // Use current directory for images

    criu_set_log_file("criu.log");
    criu_set_log_level(4);             // Log level 4 for detailed logging
    criu_set_pid(getpid());            // Set PID of the process to checkpoint
    criu_set_leave_running(1);         // Keep the process running after dump
    criu_set_service_address("/tmp/criu_service.socket");
}

int main() {
    set_criu();

    if (access("./criu_images", F_OK) == 0) {
        printf("Restoring from checkpoint...\n");
        criu_restore();
        perror("CRIU restore failed");
        exit(EXIT_FAILURE);
    }
    for (int i = 1; i <= TOTAL_ITER; i++) {
        printf("Iteration: %d\n", i);
        sleep(1);  // Simulate long-running work

        // Create a checkpoint after a certain number of iterations
        if (i == CHECKPOINT_ITER) {
            printf("Creating checkpoint...\n");

            // Perform the checkpoint
            if (criu_dump() < 0) {
                perror("CRIU checkpoint failed");
                exit(EXIT_FAILURE);
            }

            printf("Checkpoint created successfully!\n");
            printf("Exiting after creating a checkpoint, run again to restore from it.\n");
            exit(EXIT_SUCCESS);
        }
    }

    printf("Loop finished.\n");
    close(fd);
    return 0;
}
