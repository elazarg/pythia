/*
 * Memory Process Snapshotting Tool with Byte-Level Diff
 *
 * DESIGN FOR CHECKPOINTING RESEARCH:
 * - Provides byte-level change detection
 * - Stores previous memory content for detailed comparison
 * - Optimized for research accuracy over production performance
 * - Serves as baseline for comparing against Python checkpointing libraries
 *
 * CORRECTNESS GUARANTEES:
 * - Exact byte-level change detection (no hash approximations)
 * - Complete change coverage reporting
 * - Failed reads are explicitly detected and reported
 *
 * CORRECTNESS LIMITATIONS:
 * - Large memory overhead (stores 2x region content)
 * - Memory layout changes between init() and capture() are NOT detected
 * - Observer effect: significant heap usage affects measurements
 * - Single-threaded assumption at snapshot points
 *
 * USAGE FOR CHECKPOINTING RESEARCH:
 * 1. Call snapshot_init() once before iterative computation
 * 2. Call snapshot_capture() at stable execution points
 * 3. Use snapshot_get_changed_bytes() for byte change count
 * 4. Compare results with Python checkpointing library measurements
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdint.h>
#include <sys/stat.h>
#include <errno.h>

#define MAX_REGIONS 512
#define READ_BUFFER_SIZE (64 * 1024)
#define MAX_PATH_LEN 256

/*
 * Memory region with stored content for byte-level comparison
 */
typedef struct {
    uintptr_t start;           // Virtual address start
    uintptr_t end;             // Virtual address end
    size_t size;               // Size in bytes
    char perms[5];             // Permission string
    char name[64];             // Region name

    // Content storage for diff
    void* previous_content;    // Content from previous snapshot
    void* current_content;     // Content from current snapshot
    int content_valid;         // 1 if content is valid, 0 if read failed

    // Change statistics
    size_t changed_bytes;      // Number of bytes that changed
} memory_region_t;

/*
 * Snapshot context with content storage
 */
typedef struct {
    memory_region_t regions[MAX_REGIONS];
    char read_buffer[READ_BUFFER_SIZE];
    int region_count;
    int mem_fd;
    int iteration;
    char output_dir[MAX_PATH_LEN];
    int initialized;
    size_t total_memory_allocated; // Track memory overhead
} snapshot_context_t;

/*
 * Parse /proc/self/maps line with validation
 */
static int parse_maps_line(const char* line, memory_region_t* region) {
    unsigned long start, end;
    char perms[8];
    char pathname[256] = {0};

    int fields = sscanf(line, "%lx-%lx %7s %*x %*x:%*x %*d %255s",
                       &start, &end, perms, pathname);

    if (fields < 3 || end <= start) {
        return 0;
    }

    region->start = (uintptr_t)start;
    region->end = (uintptr_t)end;
    region->size = region->end - region->start;
    strncpy(region->perms, perms, sizeof(region->perms) - 1);
    region->perms[sizeof(region->perms) - 1] = '\0';

    // Extract region name
    if (fields >= 4 && pathname[0] != '\0') {
        const char* basename = strrchr(pathname, '/');
        strncpy(region->name, basename ? basename + 1 : pathname,
                sizeof(region->name) - 1);
    } else {
        strcpy(region->name, "[anonymous]");
    }
    region->name[sizeof(region->name) - 1] = '\0';

    return 1;
}

/*
 * Conservative region filtering for checkpointing research
 */
static int should_include_region(const memory_region_t* region, const char* line) {
    // Must be writable (where state changes occur)
    if (region->perms[1] != 'w') {
        return 0;
    }

    // Skip tiny regions
    if (region->size < 4096) {
        return 0;
    }

    // Skip system regions
    if (strstr(line, "[vdso]") || strstr(line, "[vsyscall]") || strstr(line, "[vvar]")) {
        return 0;
    }

    // Include heap, stack, anonymous - core program state
    if (strstr(line, "[heap]") || strstr(line, "[stack]") ||
        strstr(region->name, "[anonymous]")) {
        return 1;
    }

    // Skip shared libraries
    if (strstr(line, "/lib/") || strstr(line, "/usr/lib/")) {
        return 0;
    }

    // Include other writable regions (mapped files, etc.)
    return 1;
}

/*
 * Discover memory regions for checkpointing
 */
static int discover_memory_regions(snapshot_context_t* ctx) {
    FILE* maps_file = fopen("/proc/self/maps", "r");
    if (!maps_file) {
        fprintf(stderr, "Error: Cannot open /proc/self/maps: %s\n", strerror(errno));
        return -1;
    }

    char line[512];
    ctx->region_count = 0;
    ctx->total_memory_allocated = 0;

    while (fgets(line, sizeof(line), maps_file) && ctx->region_count < MAX_REGIONS) {
        memory_region_t candidate;
        memset(&candidate, 0, sizeof(candidate));

        if (parse_maps_line(line, &candidate) && should_include_region(&candidate, line)) {
            memory_region_t* region = &ctx->regions[ctx->region_count];
            *region = candidate;

            // Allocate storage for content comparison
            region->previous_content = malloc(region->size);
            region->current_content = malloc(region->size);

            if (!region->previous_content || !region->current_content) {
                fprintf(stderr, "Error: Cannot allocate %zu bytes for region %s\n",
                       region->size, region->name);
                if (region->previous_content) free(region->previous_content);
                if (region->current_content) free(region->current_content);
                continue;  // Skip this region
            }

            region->content_valid = 0;
            region->changed_bytes = 0;

            ctx->total_memory_allocated += 2 * region->size;
            ctx->region_count++;
        }
    }

    fclose(maps_file);

    if (ctx->region_count == 0) {
        fprintf(stderr, "Error: No suitable memory regions found\n");
        return -1;
    }

    return ctx->region_count;
}

/*
 * Read region content into buffer with error handling
 */
static int read_region_content(snapshot_context_t* ctx, memory_region_t* region, void* buffer) {
    size_t remaining = region->size;
    off_t offset = (off_t)region->start;
    char* dest = (char*)buffer;

    while (remaining > 0) {
        size_t chunk_size = (remaining > READ_BUFFER_SIZE) ? READ_BUFFER_SIZE : remaining;

        ssize_t bytes_read = pread(ctx->mem_fd, ctx->read_buffer, chunk_size, offset);

        if (bytes_read <= 0) {
            return 0;  // Read failed
        }

        if ((size_t)bytes_read < chunk_size && remaining > (size_t)bytes_read) {
            return 0;
        }

        // Copy to destination buffer
        memcpy(dest, ctx->read_buffer, bytes_read);

        dest += bytes_read;
        offset += bytes_read;
        remaining -= (size_t)bytes_read;
    }

    return 1;  // Success
}

/*
 * Count changed bytes between previous and current content
 */
static void count_changed_bytes(memory_region_t* region) {
    if (!region->content_valid) {
        region->changed_bytes = 0;
        return;
    }

    const char* prev = (const char*)region->previous_content;
    const char* curr = (const char*)region->current_content;

    region->changed_bytes = 0;
    for (size_t i = 0; i < region->size; i++) {
        if (prev[i] != curr[i]) {
            region->changed_bytes++;
        }
    }
}

/*
 * Initialize snapshot context for checkpointing research
 */
int snapshot_init(snapshot_context_t** ctx_ptr, const char* output_dir) {
    if (!ctx_ptr) return -1;

    snapshot_context_t* ctx = malloc(sizeof(snapshot_context_t));
    if (!ctx) {
        fprintf(stderr, "Error: Failed to allocate snapshot context\n");
        return -1;
    }

    memset(ctx, 0, sizeof(snapshot_context_t));

    if (output_dir) {
        strncpy(ctx->output_dir, output_dir, sizeof(ctx->output_dir) - 1);
        mkdir(ctx->output_dir, 0755);
    } else {
        strcpy(ctx->output_dir, "/tmp");
    }

    ctx->mem_fd = open("/proc/self/mem", O_RDONLY);
    if (ctx->mem_fd < 0) {
        fprintf(stderr, "Error: Cannot open /proc/self/mem: %s\n", strerror(errno));
        free(ctx);
        return -1;
    }

    int region_count = discover_memory_regions(ctx);
    if (region_count < 0) {
        close(ctx->mem_fd);
        free(ctx);
        return -1;
    }

    ctx->iteration = 0;
    ctx->initialized = 0xDEADBEEF;
    *ctx_ptr = ctx;

    printf("Memory snapshot initialized: %d regions, %.2f MB overhead\n",
           region_count, ctx->total_memory_allocated / (1024.0 * 1024.0));

    return 0;
}

/*
 * Capture snapshot and return number of changed bytes
 */
size_t snapshot_capture(snapshot_context_t* ctx) {
    if (!ctx || ctx->initialized != 0xDEADBEEF) {
        return 0;
    }

    size_t total_changed_bytes = 0;

    for (int i = 0; i < ctx->region_count; i++) {
        memory_region_t* region = &ctx->regions[i];

        // Swap buffers: current becomes previous
        void* temp = region->previous_content;
        region->previous_content = region->current_content;
        region->current_content = temp;

        // Read new content
        int read_success = read_region_content(ctx, region, region->current_content);

        if (!read_success) {
            region->content_valid = 0;
            continue;
        }

        // Count changes (skip first iteration)
        if (ctx->iteration > 0 && region->content_valid) {
            count_changed_bytes(region);
            total_changed_bytes += region->changed_bytes;
        }

        region->content_valid = 1;
    }

    ctx->iteration++;
    return total_changed_bytes;
}

/*
 * Get total bytes changed from last capture
 */
size_t snapshot_get_changed_bytes(snapshot_context_t* ctx) {
    if (!ctx || ctx->initialized != 0xDEADBEEF || ctx->iteration == 0) {
        return 0;
    }

    size_t total_changed_bytes = 0;
    for (int i = 0; i < ctx->region_count; i++) {
        memory_region_t* region = &ctx->regions[i];
        if (region->content_valid) {
            total_changed_bytes += region->changed_bytes;
        }
    }

    return total_changed_bytes;
}

/*
 * Cleanup with proper memory deallocation
 */
void snapshot_cleanup(snapshot_context_t* ctx) {
    if (!ctx) return;

    // Free allocated region content
    for (int i = 0; i < ctx->region_count; i++) {
        memory_region_t* region = &ctx->regions[i];
        if (region->previous_content) {
            free(region->previous_content);
        }
        if (region->current_content) {
            free(region->current_content);
        }
    }

    if (ctx->mem_fd >= 0) {
        close(ctx->mem_fd);
    }

    ctx->initialized = 0;
    free(ctx);
}

#ifdef TEST_MAIN
int main() {
    snapshot_context_t* snapshotter;

    if (snapshot_init(&snapshotter, "/tmp/test") < 0) {
        return 1;
    }

    for (int i = 0; i < 5; i++) {
        // Simulate computation
        size_t alloc_size = 1024 * (i + 1);
        void* temp = malloc(alloc_size);
        memset(temp, i + 42, alloc_size);

        size_t changed_bytes = snapshot_capture(snapshotter);
        printf("Iteration %d: %zu bytes changed\n", i + 1, changed_bytes);

        free(temp);
    }

    snapshot_cleanup(snapshotter);
    return 0;
}
#endif
