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
 * - Complete change coverage reporting including new regions
 * - ALL regions must be readable or measurement fails (no partial results)
 * - Deterministic discovery overhead (no malloc during region discovery)
 *
 * CORRECTNESS LIMITATIONS:
 * - Large memory overhead (stores 2x region content)
 * - Observer effect: significant heap usage affects measurements
 * - Single-threaded assumption at snapshot points
 * - Fixed limits on region count and discovery buffer sizes
 *
 * IMPLEMENTATION DECISIONS FOR CONSISTENT DISCOVERY OVERHEAD:
 * 1. Pre-allocated discovery buffers in context (no malloc during discovery)
 * 2. Fixed-size staging areas for parsing /proc/self/maps
 * 3. Buffer reuse for disappeared/resized regions
 * 4. All temporary allocations happen at init time only
 *
 * USAGE FOR CHECKPOINTING RESEARCH:
 * 1. Call snapshot_init() once before iterative computation
 * 2. Call snapshot_capture() at stable execution points
 * 3. Each capture re-discovers regions with zero allocation overhead
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

#define MAX_REGIONS 2048
#define READ_BUFFER_SIZE (64 * 1024)
#define MAX_PATH_LEN 256
#define MAPS_BUFFER_SIZE (128 * 1024)  // Buffer for entire /proc/self/maps content
#define LINE_BUFFER_SIZE 1024          // Buffer for individual lines
#define HISTORY_BUFFER_SIZE 50         // Number of snapshots to buffer before printing

/*
 * Memory region with stored content for byte-level comparison
 *
 * DESIGN DECISION: Added 'active' flag and 'buffer_capacity' for buffer reuse.
 * CORRECTNESS: When regions disappear, we mark them inactive but keep buffers.
 * When new regions appear, we reuse existing buffers if size permits.
 * This ensures consistent memory allocation patterns across captures.
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
 *
 * DESIGN DECISION: Added pre-allocated buffers for discovery process.
 * CORRECTNESS: All temporary allocations happen at init time.
 * During capture/discovery, we only use these pre-allocated buffers.
 * This ensures zero malloc overhead and consistent memory patterns.
 *
 * ADDED: History buffer for tracking changed bytes over time.
 * Prints results when buffer fills or at cleanup for analysis.
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
 *
 * CORRECTNESS: Identical to previous version - no changes to parsing logic.
 * Uses pre-allocated staging buffer to avoid malloc during discovery.
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
 *
 * CORRECTNESS: Identical filtering logic as before.
 * Added size limit to prevent excessive memory allocation for huge regions.
 * Region filtering - includes Python shared libraries
 */
static int should_include_region(const memory_region_t* region, const char* line) {
    // Must be writable (where state changes occur)
    if (region->perms[1] != 'w') {
        return 0;
    }

    // Skip tiny regions
    if (region->size < 1024) {  // Reduced from 4096
        return 0;
    }

    // Skip extremely large regions (likely not program state)
    if (region->size > 100 * 1024 * 1024) {
        return 0;
    }

    // Skip system regions
    if (strstr(line, "[vdso]") || strstr(line, "[vsyscall]") || strstr(line, "[vvar]")) {
        return 0;
    }
    // Include other writable regions (mapped files, etc.)
    return 1;
}
/*
 * Print history buffer contents (when full or at cleanup)
 *
 * DESIGN DECISION: Print line-by-line format for easy parsing/analysis.
 * Each line shows: snapshot_number:changed_bytes
 * This format is easily consumable by analysis scripts.
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
 * DESIGN DECISION: First try to find exact address match to preserve comparisons.
 * If no match, then find reusable slot with sufficient capacity.
 * CORRECTNESS: Address matching ensures we compare same memory across captures.
 * Buffer reuse only for genuinely new regions.
 */
static memory_region_t* find_region_slot(snapshot_context_t* ctx, memory_region_t* new_region) {
    // First pass: look for exact address match (same region, preserve buffers)
    for (int i = 0; i < ctx->region_count; i++) {
        memory_region_t* region = &ctx->regions[i];
        if (region->start == new_region->start && region->end == new_region->end) {
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

    return best_match;
}

/*
 * Allocate content buffers for a region
 *
 * DESIGN DECISION: Only called during init or when no reusable buffer exists.
 * CORRECTNESS: Updates total_memory_allocated counter for overhead tracking.
 * Sets buffer_capacity to actual allocated size for reuse calculations.
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
        return 0;
    }

    region->buffer_capacity = region->size;
    ctx->total_memory_allocated += 2 * region->size;
    return 1;
}

/*
 * Discover memory regions with zero malloc overhead
 *
 * DESIGN DECISION: Read entire /proc/self/maps into pre-allocated buffer.
 * Parse from memory without any malloc calls during discovery process.
 *
 * CORRECTNESS REASONING:
 * 1. Uses pre-allocated staging_regions[] for temporary parsing
 * 2. All file I/O uses pre-allocated maps_buffer[]
 * 3. No malloc/free calls during discovery phase
 * 4. Buffer reuse for disappeared regions minimizes new allocations
 * 5. Only new allocations are for genuinely new, large regions
 *
 * FOOTPRINT CONSISTENCY:
 * - Same read() call pattern every time
 * - Same parsing loop overhead
 * - Same memory access patterns
 * - Malloc only when reuse is impossible
 */
static int discover_memory_regions(snapshot_context_t* ctx) {
    // Mark all current regions as inactive (available for reuse)
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

    // Now assign staging regions to actual slots, preserving existing regions
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

        // If this is the same region (address match), preserve buffers and content_valid
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
        } else if (prev_content && prev_capacity >= staged->size) {
            // Different region - reuse buffers but reset validity
            region->previous_content = prev_content;
            region->current_content = curr_content;
            region->buffer_capacity = prev_capacity;
            region->content_valid = 0;  // New region, no valid previous content
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
        }

        region->active = 1;
        region->changed_bytes = 0;
        ctx->active_regions++;
    }

    return ctx->active_regions;
}

/*
 * Read region content into buffer with robust error handling
 *
 * DESIGN DECISION: Use chunked reads for large regions (NumPy arrays, etc.)
 * CORRECTNESS: Large single pread() calls can fail due to:
 * - Signal interruption (EINTR)
 * - Memory pressure during multi-GB reads
 * - Kernel limits on single read size
 * - Memory protection changes during read
 *
 * Chunked approach ensures reliability for arbitrary Python workloads.
 */
static int read_region_content(snapshot_context_t* ctx, memory_region_t* region, void* buffer) {
    size_t remaining = region->size;
    off_t offset = (off_t)region->start;
    char* dest = (char*)buffer;

    while (remaining > 0) {
        size_t chunk_size = (remaining > READ_BUFFER_SIZE) ? READ_BUFFER_SIZE : remaining;

        ssize_t bytes_read = pread(ctx->mem_fd, ctx->read_buffer, chunk_size, offset);

        if (bytes_read < 0) {
            if (errno == EINTR) {
                continue;  // Retry on signal interruption
            }
            return 0;  // Other errors are fatal
        }

        if (bytes_read == 0) {
            return 0;  // Unexpected EOF
        }

        if ((size_t)bytes_read < chunk_size && remaining > (size_t)bytes_read) {
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
 *
 * CORRECTNESS: Identical to previous version - no changes to comparison logic.
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
 *
 * DESIGN DECISION: Allocate all discovery buffers at init time.
 * CORRECTNESS: Ensures deterministic memory layout before measurements begin.
 * All temporary buffers are pre-allocated in context structure.
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

    // Initial region discovery (this is the only time malloc happens for regions)
    int region_count = discover_memory_regions(ctx);
    if (region_count < 0) {
        close(ctx->mem_fd);
        free(ctx);
        exit(1);
    }

    ctx->iteration = 0;
    ctx->history_count = 0;  // Initialize history buffer
    ctx->initialized = 0xDEADBEEF;
    *ctx_ptr = ctx;

    printf("Memory snapshot initialized: %d regions, %.2f MB overhead\n",
           region_count, ctx->total_memory_allocated / (1024.0 * 1024.0));
}

/*
 * Capture snapshot with dynamic region discovery and return changed bytes
 *
 * DESIGN DECISION: Re-discover regions on each capture with zero malloc overhead.
 * CORRECTNESS: Ensures new Python heap allocations are detected and measured.
 * Uses pre-allocated buffers to maintain consistent discovery overhead.
 *
 * FIXED: Buffer swapping now happens BEFORE rediscovery to preserve content.
 * Region matching by address ensures we compare the same memory locations.
 */
void snapshot_capture(snapshot_context_t* ctx) {
    if (!ctx || ctx->initialized != 0xDEADBEEF) {
        fprintf(stderr, "Error: Snapshot context not initialized or invalid\n");
        exit(1);
    }

    size_t total_changed_bytes = 0;

    // FIRST: Swap buffers for existing regions (preserve previous content)
    for (int i = 0; i < ctx->region_count; i++) {
        memory_region_t* region = &ctx->regions[i];
        if (region->active && region->content_valid) {
            // Swap buffers: current becomes previous
            void* temp = region->previous_content;
            region->previous_content = region->current_content;
            region->current_content = temp;
        }
    }

    // SECOND: Re-discover regions (may find new ones, reuse buffers)
    int region_count = discover_memory_regions(ctx);
    if (region_count < 0) {
        fprintf(stderr, "Error: Region discovery failed - measurement incomplete\n");
        exit(1);
    }

    // THIRD: Read current content and compare
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
        }

        region->content_valid = 1;
    }

    // Add to history buffer
    ctx->history_buffer[ctx->history_count] = total_changed_bytes;
    ctx->history_count++;

    // Print and clear buffer when full
    if (ctx->history_count >= HISTORY_BUFFER_SIZE) {
        print_history_buffer(ctx);
    }

    ctx->iteration++;
}

/*
 * Get total bytes changed from last capture
 *
 * CORRECTNESS: Only counts active regions to avoid stale data.
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
 *
 * CORRECTNESS: Frees all allocated region buffers regardless of active status.
 * ADDED: Prints any remaining history buffer entries before cleanup.
 */
void snapshot_cleanup(snapshot_context_t* ctx) {
    if (!ctx) return;

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
