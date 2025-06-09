/*
 * Memory Process Snapshotting Tool with Byte-Level Diff and Conditional Debug
 *
 * DESIGN FOR CHECKPOINTING RESEARCH:
 * - Provides byte-level change detection
 * - Stores previous memory content for detailed comparison
 * - Optimized for research accuracy over production performance
 * - Serves as baseline for comparing against Python checkpointing libraries
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
#define READ_BUFFER_SIZE (64 * 1024)
#define MAX_PATH_LEN 256
#define MAPS_BUFFER_SIZE (128 * 1024)  // Buffer for entire /proc/self/maps content
#define LINE_BUFFER_SIZE 1024          // Buffer for individual lines
#define HISTORY_BUFFER_SIZE 50         // Number of snapshots to buffer before printing

// Debug control - set via environment variable SNAPSHOT_DEBUG
static int debug_enabled = -1;  // -1 = not initialized, 0 = disabled, 1 = enabled

/*
 * Conditional debug printing
 * Only prints if SNAPSHOT_DEBUG environment variable is set
 */
static void debug(const char* fmt, ...) {
    // Initialize debug flag on first call
    if (debug_enabled == -1) {
        const char* env_debug = getenv("SNAPSHOT_DEBUG");
        if (env_debug) {
            debug_enabled = (strcmp(env_debug, "0") != 0 &&
                           strcmp(env_debug, "") != 0 &&
                           strcmp(env_debug, "false") != 0 &&
                           strcmp(env_debug, "FALSE") != 0 &&
                           strcmp(env_debug, "NO") != 0 &&
                           strcmp(env_debug, "no") != 0) ? 1 : 0;
        } else {
            debug_enabled = 0;  // Not set = disabled
        }
    }

    if (!debug_enabled) {
        return;
    }

    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "DEBUG: ");
    vfprintf(stderr, fmt, args);
    va_end(args);
}


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
    int active;                // 1 if region exists, 0 if available for reuse
    size_t buffer_capacity;    // Actual allocated size (may be larger than size)

    // Change statistics
    size_t changed_bytes;      // Number of bytes that changed
} memory_region_t;

/*
 * Snapshot context with content storage
 */
typedef struct {
    memory_region_t regions[MAX_REGIONS];
    char read_buffer[READ_BUFFER_SIZE];
    int region_count;          // Total slots (some may be inactive)
    int active_regions;        // Currently active regions
    int mem_fd;
    int iteration;
    int initialized;
    size_t total_memory_allocated; // Track memory overhead

    // Pre-allocated discovery buffers (ZERO MALLOC during discovery)
    char maps_buffer[MAPS_BUFFER_SIZE];    // Entire /proc/self/maps content
    char line_buffer[LINE_BUFFER_SIZE];    // Current line being parsed
    memory_region_t staging_regions[MAX_REGIONS]; // Temporary parsing area

    // History tracking for analysis
    size_t history_buffer[HISTORY_BUFFER_SIZE];  // Changed bytes per snapshot
    int history_count;                           // Number of entries in buffer
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
        debug("Excluding non-writable region: %s\n", line);
        return 0;
    }

    // Skip system regions that are definitely not program state
    if (strstr(line, "[vdso]") || strstr(line, "[vsyscall]") || strstr(line, "[vvar]")) {
        debug("Excluding system region: %s\n", line);
        return 0;
    }
    return 1;

    // Always include heap and stack - core program state
    if (strstr(line, "[heap]") || strstr(line, "[stack]")) {
        debug("Including core region: %s (size: %zu KB)\n", region->name, region->size / 1024);
        return 1;
    }

    // Include anonymous regions up to reasonable limits
    if (strstr(region->name, "[anonymous]") || strstr(line, "00:00 0")) {
        // For anonymous regions, be more generous with size limits
        // Python can allocate very large memory arenas
        if (region->size > 1024 * 1024 * 1024) {  // 1GB limit instead of 100MB
            debug("Excluding huge anonymous region (%zu MB): %s\n",
                  region->size / (1024*1024), line);
            return 0;
        }

        debug("Including anonymous region: %s (size: %zu KB)\n",
              region->name, region->size / 1024);
        return 1;
    }

    // Include Python-related shared libraries
    if (strstr(line, "python") || strstr(line, "libpython") ||
        strstr(line, ".cpython-") || strstr(line, "site-packages")) {
        debug("Including Python library: %s (size: %zu KB)\n",
              region->name, region->size / 1024);
        return 1;
    }

    // Include your snapshot library (useful for debugging)
    if (strstr(line, "snapshot.so")) {
        debug("Including snapshot library: %s (size: %zu KB)\n",
              region->name, region->size / 1024);
        return 1;
    }

    // Exclude other shared libraries - they're less likely to contain changing program state
    if (strstr(line, "/lib/") || strstr(line, "/usr/lib/") || strstr(line, ".so")) {
        debug("Excluding shared library: %s\n", line);
        return 0;
    }

    // Skip tiny regions (likely not significant)
    if (region->size < 4096) {
        debug("Excluding tiny region (%zu bytes): %s\n", region->size, line);
        return 0;
    }

    // Include other potentially interesting writable regions
    debug("Including other writable region: %s (size: %zu KB)\n",
          region->name, region->size / 1024);
    return 1;
}

/*
 * Print history buffer contents (when full or at cleanup)
 */
static void print_history_buffer(snapshot_context_t* ctx) {
    if (ctx->history_count == 0) {
        return;
    }

    printf("# Snapshot history (%d entries):\n", ctx->history_count);
    int start_iteration = ctx->iteration - ctx->history_count;

    for (int i = 0; i < ctx->history_count; i++) {
        printf("%d:%zu\n", start_iteration + i + 1, ctx->history_buffer[i]);
    }

    ctx->history_count = 0;  // Clear buffer after printing
}

/*
 * Find region slot for reuse or exact matching
 */
static memory_region_t* find_region_slot(snapshot_context_t* ctx, memory_region_t* new_region) {
    // First pass: look for exact address match (same region, preserve buffers)
    for (int i = 0; i < ctx->region_count; i++) {
        memory_region_t* region = &ctx->regions[i];
        if (region->start == new_region->start && region->end == new_region->end) {
            debug("Found exact match for region 0x%lx-0x%lx (%s)\n",
                  new_region->start, new_region->end, new_region->name);
            return region;  // Found same region, keep existing buffers
        }
    }

    // Second pass: look for reusable inactive slot
    memory_region_t* best_match = NULL;
    size_t best_capacity = SIZE_MAX;

    for (int i = 0; i < ctx->region_count; i++) {
        memory_region_t* region = &ctx->regions[i];

        // Must be inactive and have sufficient capacity
        if (!region->active && region->buffer_capacity >= new_region->size) {
            // Prefer smallest buffer that fits (minimize waste)
            if (region->buffer_capacity < best_capacity) {
                best_match = region;
                best_capacity = region->buffer_capacity;
            }
        }
    }

    if (best_match) {
        debug("Reusing buffer slot for region 0x%lx-0x%lx (%s), capacity %zu\n",
              new_region->start, new_region->end, new_region->name, best_capacity);
    } else {
        debug("Need new slot for region 0x%lx-0x%lx (%s)\n",
              new_region->start, new_region->end, new_region->name);
    }

    return best_match;
}

/*
 * Allocate content buffers for a region
 */
static int allocate_region_buffers(snapshot_context_t* ctx, memory_region_t* region) {
    region->previous_content = malloc(region->size);
    region->current_content = malloc(region->size);

    if (!region->previous_content || !region->current_content) {
        if (region->previous_content) free(region->previous_content);
        if (region->current_content) free(region->current_content);
        region->previous_content = NULL;
        region->current_content = NULL;
        region->buffer_capacity = 0;
        debug("Failed to allocate buffers for region %s (%zu bytes)\n",
              region->name, region->size);
        return 0;
    }

    region->buffer_capacity = region->size;
    ctx->total_memory_allocated += 2 * region->size;
    debug("Allocated buffers for region %s (%zu bytes)\n", region->name, region->size);
    return 1;
}

/*
 * Discover memory regions with zero malloc overhead
 */
static int discover_memory_regions(snapshot_context_t* ctx) {
    debug("Starting region discovery (iteration %d)\n", ctx->iteration);

    // Mark all current regions as inactive (available for reuse)
    int previously_active = ctx->active_regions;
    for (int i = 0; i < ctx->region_count; i++) {
        ctx->regions[i].active = 0;
    }
    ctx->active_regions = 0;

    // Read entire /proc/self/maps into pre-allocated buffer
    int fd = open("/proc/self/maps", O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Error: Cannot open /proc/self/maps: %s\n", strerror(errno));
        return -1;
    }

    ssize_t bytes_read = read(fd, ctx->maps_buffer, sizeof(ctx->maps_buffer) - 1);
    close(fd);

    if (bytes_read <= 0) {
        fprintf(stderr, "Error: Cannot read /proc/self/maps\n");
        return -1;
    }

    ctx->maps_buffer[bytes_read] = '\0';

    // Parse regions using pre-allocated staging area
    int staging_count = 0;
    char* line_start = ctx->maps_buffer;
    char* line_end;

    while (line_start < ctx->maps_buffer + bytes_read && staging_count < MAX_REGIONS) {
        // Find end of current line
        line_end = strchr(line_start, '\n');
        if (!line_end) {
            line_end = ctx->maps_buffer + bytes_read;
        }

        // Copy line to parsing buffer
        size_t line_len = line_end - line_start;
        if (line_len >= sizeof(ctx->line_buffer)) {
            line_len = sizeof(ctx->line_buffer) - 1;
        }

        memcpy(ctx->line_buffer, line_start, line_len);
        ctx->line_buffer[line_len] = '\0';

        // Parse line into staging area
        memory_region_t candidate;
        memset(&candidate, 0, sizeof(candidate));

        if (parse_maps_line(ctx->line_buffer, &candidate) &&
            should_include_region(&candidate, ctx->line_buffer)) {

            ctx->staging_regions[staging_count] = candidate;
            staging_count++;
        }

        // Move to next line
        line_start = line_end + 1;
    }

    debug("Parsed %d candidate regions from /proc/self/maps\n", staging_count);

    // Now assign staging regions to actual slots, preserving existing regions
    int exact_matches = 0, reused_slots = 0, new_allocations = 0;

    for (int i = 0; i < staging_count; i++) {
        memory_region_t* staged = &ctx->staging_regions[i];

        // Try to find existing region or reusable slot
        memory_region_t* region = find_region_slot(ctx, staged);

        if (!region) {
            // Need new slot
            if (ctx->region_count >= MAX_REGIONS) {
                fprintf(stderr, "Warning: Maximum regions exceeded, skipping region %s\n",
                       staged->name);
                continue;
            }
            region = &ctx->regions[ctx->region_count];
            memset(region, 0, sizeof(*region));
            ctx->region_count++;
        }

        // Check if this is the same region (address match)
        int same_region = (region->start == staged->start && region->end == staged->end);
        void* prev_content = region->previous_content;
        void* curr_content = region->current_content;
        size_t prev_capacity = region->buffer_capacity;
        int was_valid = region->content_valid;

        // Copy region info
        *region = *staged;

        if (same_region && prev_content) {
            // Same region - preserve existing buffers and validity
            region->previous_content = prev_content;
            region->current_content = curr_content;
            region->buffer_capacity = prev_capacity;
            region->content_valid = was_valid;
            exact_matches++;
        } else if (prev_content && prev_capacity >= staged->size) {
            // Different region - reuse buffers but reset validity
            region->previous_content = prev_content;
            region->current_content = curr_content;
            region->buffer_capacity = prev_capacity;
            region->content_valid = 0;  // New region, no valid previous content
            reused_slots++;
        } else {
            // Need new allocation
            if (prev_content) {
                free(prev_content);
                free(curr_content);
                ctx->total_memory_allocated -= 2 * prev_capacity;
            }

            if (!allocate_region_buffers(ctx, region)) {
                fprintf(stderr, "Error: Cannot allocate buffers for region %s - measurement accuracy compromised\n",
                       region->name);
                return -1;  // Fail completely rather than provide inaccurate results
            }
            region->content_valid = 0;
            new_allocations++;
        }

        region->active = 1;
        region->changed_bytes = 0;
        ctx->active_regions++;
    }

    debug("Region assignment: %d exact matches, %d reused slots, %d new allocations\n",
          exact_matches, reused_slots, new_allocations);
    debug("Active regions: %d (was %d)\n", ctx->active_regions, previously_active);

    return ctx->active_regions;
}

/*
 * Read region content into buffer with robust error handling
 */
static int read_region_content(snapshot_context_t* ctx, memory_region_t* region, void* buffer) {
    size_t remaining = region->size;
    off_t offset = (off_t)region->start;
    char* dest = (char*)buffer;

    debug("Reading region %s (0x%lx, %zu bytes)\n", region->name, region->start, region->size);

    while (remaining > 0) {
        size_t chunk_size = (remaining > READ_BUFFER_SIZE) ? READ_BUFFER_SIZE : remaining;

        ssize_t bytes_read = pread(ctx->mem_fd, ctx->read_buffer, chunk_size, offset);

        if (bytes_read < 0) {
            if (errno == EINTR) {
                continue;  // Retry on signal interruption
            }
            debug("Read failed for region %s: %s\n", region->name, strerror(errno));
            return 0;  // Other errors are fatal
        }

        if (bytes_read == 0) {
            debug("Unexpected EOF for region %s\n", region->name);
            return 0;  // Unexpected EOF
        }

        if ((size_t)bytes_read < chunk_size && remaining > (size_t)bytes_read) {
            debug("Short read for region %s: got %zd, expected %zu\n",
                  region->name, bytes_read, chunk_size);
            return 0;  // Short read when more data expected
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

    debug("Region %s: %zu bytes changed\n", region->name, region->changed_bytes);
}

/*
 * Initialize snapshot context for checkpointing research
 */
void snapshot_init(snapshot_context_t** ctx_ptr) {
    if (!ctx_ptr) exit(1);

    snapshot_context_t* ctx = malloc(sizeof(snapshot_context_t));
    if (!ctx) {
        fprintf(stderr, "Error: Failed to allocate snapshot context\n");
        exit(1);
    }

    memset(ctx, 0, sizeof(snapshot_context_t));

    ctx->mem_fd = open("/proc/self/mem", O_RDONLY);
    if (ctx->mem_fd < 0) {
        fprintf(stderr, "Error: Cannot open /proc/self/mem: %s\n", strerror(errno));
        free(ctx);
        exit(1);
    }

    debug("Snapshot context initialized, discovering initial regions\n");

    // Initial region discovery (this is the only time malloc happens for regions)
    int region_count = discover_memory_regions(ctx);
    if (region_count < 0) {
        close(ctx->mem_fd);
        free(ctx);
        exit(1);
    }

    ctx->iteration = 0;
    ctx->history_count = 0;
    ctx->initialized = 0xDEADBEEF;
    *ctx_ptr = ctx;

    printf("Memory snapshot initialized: %d regions, %.2f MB overhead\n",
           region_count, ctx->total_memory_allocated / (1024.0 * 1024.0));
}

/*
 * Capture snapshot with dynamic region discovery and return changed bytes
 */
void snapshot_capture(snapshot_context_t* ctx) {
    if (!ctx || ctx->initialized != 0xDEADBEEF) {
        fprintf(stderr, "Error: Snapshot context not initialized or invalid\n");
        exit(1);
    }

    debug("=== CAPTURE START (iteration %d) ===\n", ctx->iteration);
    size_t total_changed_bytes = 0;

    // FIRST: Swap buffers for existing regions (preserve previous content)
    int swapped_count = 0;
    for (int i = 0; i < ctx->region_count; i++) {
        memory_region_t* region = &ctx->regions[i];
        if (region->active && region->content_valid) {
            // Swap buffers: current becomes previous
            void* temp = region->previous_content;
            region->previous_content = region->current_content;
            region->current_content = temp;
            swapped_count++;
        }
    }

    debug("Swapped buffers for %d regions\n", swapped_count);

    // SECOND: Re-discover regions (may find new ones, reuse buffers)
    int region_count = discover_memory_regions(ctx);
    if (region_count < 0) {
        fprintf(stderr, "Error: Region discovery failed - measurement incomplete\n");
        exit(1);
    }

    // THIRD: Read current content and compare
    int regions_with_changes = 0;
    for (int i = 0; i < ctx->region_count; i++) {
        memory_region_t* region = &ctx->regions[i];

        if (!region->active) {
            continue;  // Skip inactive slots
        }

        // Read new content into current_content buffer
        int read_success = read_region_content(ctx, region, region->current_content);

        if (!read_success) {
            fprintf(stderr, "Error: Cannot read region %s at 0x%lx - measurement incomplete\n",
                   region->name, region->start);
            exit(1);
        }

        // Count changes (skip first iteration)
        if (ctx->iteration > 0 && region->content_valid) {
            count_changed_bytes(region);
            total_changed_bytes += region->changed_bytes;
            if (region->changed_bytes > 0) {
                regions_with_changes++;
            }
        }

        region->content_valid = 1;
    }

    debug("Iteration %d: %d regions with changes, %zu total bytes changed\n",
          ctx->iteration, regions_with_changes, total_changed_bytes);

    // Add to history buffer
    ctx->history_buffer[ctx->history_count] = total_changed_bytes;
    ctx->history_count++;

    // Print and clear buffer when full
    if (ctx->history_count >= HISTORY_BUFFER_SIZE) {
        print_history_buffer(ctx);
    }

    ctx->iteration++;
    debug("=== CAPTURE END ===\n");
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
        if (region->active && region->content_valid) {
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

    debug("Cleaning up snapshot context\n");

    // Print any remaining history entries
    print_history_buffer(ctx);

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

    snapshot_init(&snapshotter);

    for (int i = 0; i < 5; i++) {
        // Simulate computation with varying allocation patterns
        size_t alloc_size = 1024 * (i + 1);
        void* temp = malloc(alloc_size);
        memset(temp, i + 42, alloc_size);

        snapshot_capture(snapshotter);

        free(temp);
    }

    snapshot_cleanup(snapshotter);
    return 0;
}
#endif
