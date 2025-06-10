/*
 * Capture complete memory snapshot to numbered subdirectory
 *
 * SNAPSHOT PROCESS:
 * 1. Scan /proc/self/maps for current writable regions
 * 2. Create subdirectory: base_dir/<counter>/
 * 3. Create separate file per region: region_<start_addr>.bin
 * 4. Create regions.txt: human-readable region metadata
 * 5. Print subdirectory path to stdout for external processing
 * 6. Increment/*
 * Memory Process Snapshotting Tool with Sparse File Output
 *
 * PURPOSE:
 * Captures complete memory snapshots of a running process for research analysis.
 * Designed for studying memory changes in Python/NumPy programs where traditional
 * memory profilers are too heavyweight or intrusive.
 *
 * HOW IT WORKS:
 * 1. Scans /proc/self/maps to find all writable memory regions (heap, stack, etc.)
 * 2. Reads each region directly from /proc/self/mem
 * 3. Writes to sparse files where file_offset = memory_address
 * 4. Creates numbered subdirectories for easy sequential comparison
 *
 * SPARSE FILE FORMAT:
 * - Physical address 0x7fff12340000 maps to file offset 0x7fff12340000
 * - Unmapped address ranges become sparse holes (consume no disk space)
 * - External tools can seek to any address and read the memory content
 *
 * DIRECTORY STRUCTURE:
 * base_dir/
 *   0/memory.sparse    # First snapshot (complete process memory)
 *   0/regions.txt      # Memory map metadata
 *   1/memory.sparse    # Second snapshot
 *   1/regions.txt
 *   ...
 *
 * USAGE:
 *   snapshot_init("./snapshots");
 *   snapshot_capture();  // Creates ./snapshots/0/, prints "./snapshots/0"
 *   // ... modify memory ...
 *   snapshot_capture();  // Creates ./snapshots/1/, prints "./snapshots/1"
 *   snapshot_cleanup();
 *
 * EXTERNAL PROCESSING:
 * Compare snapshots with tools like:
 *   cmp -l snapshots/0/memory.sparse snapshots/1/memory.sparse
 *   # Or custom tools that read sparse files at specific addresses
 *
 * RESEARCH DESIGN PHILOSOPHY:
 * - Zero memory overhead (no double-buffering)
 * - Fail-fast on errors (exit rather than complex error handling)
 * - Simple API suitable for automated testing
 * - External processing for flexibility
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdint.h>
#include <sys/stat.h>
#include <errno.h>
#include <stdarg.h>

#define MAX_REGIONS 2048
#define READ_BUFFER_SIZE (1024 * 1024)  // 1MB read buffer

// Global context - research tool simplicity
static struct {
    char base_dir[256];
    int snapshot_counter;
    int mem_fd;
    char read_buffer[READ_BUFFER_SIZE];

    struct {
        uintptr_t start, end;
        size_t size;
        char perms[5];
        char name[64];
    } regions[MAX_REGIONS];
    int region_count;
} ctx;

// Debug control via environment variable
static int debug_enabled = -1;

static void debug(const char* fmt, ...) {
    if (debug_enabled == -1) {
        debug_enabled = getenv("SNAPSHOT_DEBUG") ? 1 : 0;
    }
    if (!debug_enabled) return;

    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "DEBUG: ");
    vfprintf(stderr, fmt, args);
    va_end(args);
}

/*
 * Parse /proc/self/maps line into region structure
 *
 * MAPS FORMAT: "start-end perms offset dev:inode path"
 * Example: "7fff12340000-7fff12350000 rw-p 00000000 00:00 0 [stack]"
 *
 * Extracts: start address, end address, permissions, and region name
 * Returns: 1 if parsed successfully, 0 if line is invalid
 */
static int parse_maps_line(const char* line, int region_idx) {
    unsigned long start, end;
    char perms[8];
    char pathname[256] = {0};

    int fields = sscanf(line, "%lx-%lx %7s %*x %*x:%*x %*d %255s",
                       &start, &end, perms, pathname);

    if (fields < 3 || end <= start) {
        return 0;
    }

    ctx.regions[region_idx].start = start;
    ctx.regions[region_idx].end = end;
    ctx.regions[region_idx].size = end - start;
    strncpy(ctx.regions[region_idx].perms, perms, 4);
    ctx.regions[region_idx].perms[4] = '\0';

    // Extract region name
    if (fields >= 4 && pathname[0]) {
        const char* basename = strrchr(pathname, '/');
        strncpy(ctx.regions[region_idx].name, basename ? basename + 1 : pathname, 63);
    } else {
        strcpy(ctx.regions[region_idx].name, "[anonymous]");
    }
    ctx.regions[region_idx].name[63] = '\0';

    return 1;
}

/*
 * Check if region should be included in snapshot
 *
 * INCLUSION CRITERIA:
 * - Must be writable (perms[1] == 'w')
 * - Excludes system regions: [vdso], [vsyscall], [vvar]
 *
 * RATIONALE:
 * Only writable regions can contain program state changes.
 * System regions contain kernel-provided read-only data.
 */
static int should_include_region(int region_idx) {
    // Only writable regions
    if (ctx.regions[region_idx].perms[1] != 'w') {
        return 0;
    }

    // Skip system regions
    const char* name = ctx.regions[region_idx].name;
    if (strstr(name, "[vdso]") || strstr(name, "[vsyscall]") || strstr(name, "[vvar]")) {
        return 0;
    }

    return 1;
}

/*
 * Discover writable memory regions from /proc/self/maps
 *
 * PROCESS:
 * 1. Read /proc/self/maps line by line
 * 2. Parse each line to extract memory region info
 * 3. Filter to include only writable regions
 * 4. Store in global ctx.regions array
 *
 * LIMITATIONS:
 * - MAX_REGIONS limit (2048) - warns if exceeded
 * - Exits on /proc/self/maps read failure
 */
static void discover_regions() {
    FILE* maps = fopen("/proc/self/maps", "r");
    if (!maps) {
        fprintf(stderr, "Error: Cannot open /proc/self/maps: %s\n", strerror(errno));
        exit(1);
    }

    char line[512];
    ctx.region_count = 0;

    while (fgets(line, sizeof(line), maps) && ctx.region_count < MAX_REGIONS) {
        if (parse_maps_line(line, ctx.region_count) &&
            should_include_region(ctx.region_count)) {

            debug("Including region: %s (size: %zu MB)\n",
                  ctx.regions[ctx.region_count].name,
                  ctx.regions[ctx.region_count].size / (1024*1024));
            ctx.region_count++;
        }
    }

    fclose(maps);

    if (ctx.region_count == MAX_REGIONS) {
        fprintf(stderr, "Warning: Hit MAX_REGIONS limit (%d), some regions may be missing\n", MAX_REGIONS);
    }

    debug("Discovered %d writable regions\n", ctx.region_count);
}

/*
 * Copy memory region to separate file named by address
 *
 * SEPARATE FILE APPROACH:
 * - Each region gets its own file: region_<start_addr>.bin
 * - File contains raw memory bytes starting from offset 0
 * - External tools use regions.txt to map files to addresses
 * - Avoids filesystem limits with large sparse file offsets
 *
 * FILENAME FORMAT: "region_<hex_address>.bin"
 * Example: "region_55ecaecc4000.bin" for address 0x55ecaecc4000
 */
static void copy_region_to_separate_file(const char* snapshot_dir, int region_idx) {
    uintptr_t start = ctx.regions[region_idx].start;
    size_t size = ctx.regions[region_idx].size;

    debug("Copying region %s (0x%lx, %zu bytes) to separate file\n",
          ctx.regions[region_idx].name, start, size);

    // Create filename based on start address
    char region_file[600];
    snprintf(region_file, sizeof(region_file), "%s/region_%lx.bin", snapshot_dir, start);

    int region_fd = open(region_file, O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (region_fd < 0) {
        fprintf(stderr, "Error: Cannot create region file %s: %s\n", region_file, strerror(errno));
        exit(1);
    }

    // Copy region data starting from file offset 0
    size_t bytes_remaining = size;
    uintptr_t current_addr = start;

    while (bytes_remaining > 0) {
        size_t chunk_size = (bytes_remaining < READ_BUFFER_SIZE) ? bytes_remaining : READ_BUFFER_SIZE;

        ssize_t bytes_read = pread(ctx.mem_fd, ctx.read_buffer, chunk_size, current_addr);
        if (bytes_read <= 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "Error: Cannot read memory at 0x%lx: %s\n", current_addr, strerror(errno));
            close(region_fd);
            exit(1);
        }

        ssize_t bytes_written = write(region_fd, ctx.read_buffer, bytes_read);
        if (bytes_written != bytes_read) {
            fprintf(stderr, "Error: Cannot write to region file %s: %s\n", region_file, strerror(errno));
            close(region_fd);
            exit(1);
        }

        current_addr += bytes_read;
        bytes_remaining -= bytes_read;
    }

    close(region_fd);
    debug("Successfully wrote %zu bytes to %s\n", size, region_file);
}

/*
 * Write region metadata to regions.txt file
 *
 * FORMAT: "start_addr end_addr size permissions name filename"
 * Example: "0x00007fff12340000 0x00007fff12350000 65536 rw-p [stack] region_7fff12340000.bin"
 *
 * PURPOSE:
 * Maps region filenames to memory addresses and provides metadata.
 * External tools use this to understand which file contains which memory region.
 */
static void write_region_info(int info_fd, int region_idx) {
    char line[384];
    int len = snprintf(line, sizeof(line), "0x%016lx 0x%016lx %zu %s %s region_%lx.bin\n",
                      ctx.regions[region_idx].start,
                      ctx.regions[region_idx].end,
                      ctx.regions[region_idx].size,
                      ctx.regions[region_idx].perms,
                      ctx.regions[region_idx].name,
                      ctx.regions[region_idx].start);

    write(info_fd, line, len);
}

/*
 * Initialize snapshot system with base directory
 *
 * SETUP:
 * - Creates base directory if it doesn't exist
 * - Opens /proc/self/mem for reading process memory
 * - Initializes global context and snapshot counter
 *
 * REQUIREMENTS:
 * - Process must have permission to read its own /proc/self/mem
 * - Base directory must be writable
 *
 * ERRORS:
 * Exits on NULL base_dir, directory creation failure, or /proc/self/mem access failure
 */
void snapshot_init(const char* base_dir) {
    if (!base_dir) {
        fprintf(stderr, "Error: base_dir cannot be NULL\n");
        exit(1);
    }

    strncpy(ctx.base_dir, base_dir, sizeof(ctx.base_dir) - 1);
    ctx.base_dir[sizeof(ctx.base_dir) - 1] = '\0';

    // Create base directory
    if (mkdir(ctx.base_dir, 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "Error: Cannot create directory %s: %s\n", ctx.base_dir, strerror(errno));
        exit(1);
    }

    // Open /proc/self/mem
    ctx.mem_fd = open("/proc/self/mem", O_RDONLY);
    if (ctx.mem_fd < 0) {
        fprintf(stderr, "Error: Cannot open /proc/self/mem: %s\n", strerror(errno));
        exit(1);
    }

    ctx.snapshot_counter = 0;
    debug("Snapshot initialized: %s\n", ctx.base_dir);
}

/*
 * Capture complete memory snapshot to numbered subdirectory
 *
 * SNAPSHOT PROCESS:
 * 1. Scan /proc/self/maps for current writable regions
 * 2. Create subdirectory: base_dir/<counter>/
 * 3. Create separate file per region: region_<start_addr>.bin
 * 4. Create regions.txt: human-readable region metadata
 * 5. Print subdirectory path to stdout for external processing
 * 6. Increment counter for next snapshot
 *
 * OUTPUT FILES:
 * - region_<addr>.bin: Raw memory dump of each region (starts at file offset 0)
 * - regions.txt: Maps filenames to memory addresses and metadata
 *
 * EXTERNAL INTERFACE:
 * Prints subdirectory path (e.g., "./snapshots/0") to stdout.
 * External tools can parse this output to locate snapshot files.
 *
 * ATOMICITY:
 * Memory regions are discovered fresh each time to handle
 * dynamic allocation (malloc/free) between snapshots.
 *
 * SEPARATE FILES APPROACH:
 * Avoids filesystem limits with large sparse file offsets.
 * Each region becomes a regular file containing raw memory bytes.
 */
void snapshot_capture() {
    debug("=== Snapshot %d ===\n", ctx.snapshot_counter);

    discover_regions();

    // Create snapshot directory
    char snapshot_dir[384];
    snprintf(snapshot_dir, sizeof(snapshot_dir), "%s/%d", ctx.base_dir, ctx.snapshot_counter);

    if (mkdir(snapshot_dir, 0755) != 0) {
        fprintf(stderr, "Error: Cannot create %s: %s\n", snapshot_dir, strerror(errno));
        exit(1);
    }

    // Create regions info file
    char regions_file[512];
    snprintf(regions_file, sizeof(regions_file), "%s/regions.txt", snapshot_dir);

    int info_fd = open(regions_file, O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (info_fd < 0) {
        fprintf(stderr, "Error: Cannot create regions file %s: %s\n", regions_file, strerror(errno));
        exit(1);
    }

    // Dump all regions to separate files
    size_t total_bytes = 0;
    for (int i = 0; i < ctx.region_count; i++) {
        write_region_info(info_fd, i);
        copy_region_to_separate_file(snapshot_dir, i);
        total_bytes += ctx.regions[i].size;
    }

    close(info_fd);

    debug("Captured %d regions, %zu MB total\n", ctx.region_count, total_bytes / (1024*1024));

    // Output for external processor
    printf("%s\n", snapshot_dir);
    fflush(stdout);

    ctx.snapshot_counter++;
}

/*
 * Cleanup snapshot resources
 *
 * Closes /proc/self/mem file descriptor and resets global state.
 * Safe to call multiple times.
 */
void snapshot_cleanup() {
    if (ctx.mem_fd >= 0) {
        close(ctx.mem_fd);
        ctx.mem_fd = -1;
    }
    debug("Cleanup complete\n");
}

#ifdef TEST_SNAPSHOT
int main() {
    snapshot_init("./test_snapshots");

    printf("# Testing snapshot tool\n");

    for (int i = 0; i < 3; i++) {
        size_t alloc_size = 1024 * 1024 * (i + 1);
        void* temp = malloc(alloc_size);
        memset(temp, i + 42, alloc_size);

        printf("# Iteration %d (allocated %zu MB)\n", i + 1, alloc_size / (1024 * 1024));
        snapshot_capture();

        free(temp);
    }

    snapshot_cleanup();
    printf("# Test completed\n");
    return 0;
}
#endif
