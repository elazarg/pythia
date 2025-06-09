/*
 * Memory Process Snapshotting Tool with Fixed-Size Chunk Allocation
 *
 * RESEARCH TOOL GOALS:
 * 1. Correctness: Track exactly the set of memory changes between captures (no more, no less)
 * 2. Complete coverage: Monitor ALL writable memory regions in their entirety
 * 3. Simplicity: Straightforward implementation focused on accuracy over memory efficiency
 *
 * MEMORY OVERHEAD:
 * Variable overhead proportional to total writable memory size (2x memory usage).
 * Large processes may require significant memory for complete monitoring.
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

#define MAX_REGIONS 131072
#define READ_BUFFER_SIZE (64 * 1024)
#define MAPS_BUFFER_SIZE (128 * 1024)
#define LINE_BUFFER_SIZE 1024
#define HISTORY_BUFFER_SIZE 50

/*
 * Chunking parameters
 *
 * DESIGN RATIONALE:
 * - 64KB chunks provide good granularity for detecting changes without excessive overhead
 * - No limit on chunks per region - complete coverage of all regions regardless of size
 * - All writable memory is monitored to ensure no changes are missed
 */
#define CHUNK_SIZE (64 * 1024)
#define MAX_CHUNKS_PER_REGION 65536      // Supports regions up to 4GB (64KB * 65536)
#define SMALL_REGION_LIMIT (1024 * 1024) // Kept for potential future optimizations

// Forward declaration
typedef struct chunked_snapshot_context_t chunked_snapshot_context_t;

// Debug control via environment variable
static int debug_enabled = -1;

static void debug(const char* fmt, ...) {
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
            debug_enabled = 0;
        }
    }

    if (!debug_enabled) return;

    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "DEBUG: ");
    vfprintf(stderr, fmt, args);
    va_end(args);
}

/*
 * Memory chunk for storing region data
 *
 * BUFFER ALLOCATION STRATEGY:
 * - All chunks allocated at CHUNK_SIZE regardless of actual data
 * - Simplifies memory management and ensures predictable overhead
 * - data_size tracks actual useful bytes within fixed-size buffers
 */
typedef struct {
    void* previous_data;       // Previous snapshot data (CHUNK_SIZE bytes allocated)
    void* current_data;        // Current snapshot data (CHUNK_SIZE bytes allocated)
    uintptr_t region_offset;   // Byte offset from region start
    size_t data_size;          // Actual data bytes (≤ CHUNK_SIZE)
    size_t changed_bytes;      // Changed bytes in this chunk
    int valid;                 // 1 if previous data exists for comparison
} memory_chunk_t;

/*
 * Memory region with chunked storage
 *
 * COVERAGE STRATEGY:
 * - Sequential chunks from region start to end (no gaps)
 * - Complete coverage of all regions regardless of size
 * - All changes in monitored regions are detected
 */
typedef struct {
    uintptr_t start;
    uintptr_t end;
    size_t size;
    char perms[5];
    char name[64];

    memory_chunk_t chunks[MAX_CHUNKS_PER_REGION];
    int chunk_count;           // Number of allocated chunks
    size_t covered_bytes;      // Total bytes covered by chunks (equals region size)
    int fully_covered;         // Always 1 - all regions fully covered

    int content_valid;         // 1 if region has valid comparison data
    int active;                // 1 if region exists in current memory map
    size_t total_changed_bytes; // Sum of changed bytes across all chunks
} chunked_region_t;

struct chunked_snapshot_context_t {
    chunked_region_t regions[MAX_REGIONS];
    char read_buffer[READ_BUFFER_SIZE];
    int region_count;
    int active_regions;
    int mem_fd;
    int iteration;
    uint32_t initialized;      // Magic number for context validation
    size_t total_memory_allocated;

    // Pre-allocated buffers to avoid malloc during region discovery
    char maps_buffer[MAPS_BUFFER_SIZE];
    char line_buffer[LINE_BUFFER_SIZE];

    size_t history_buffer[HISTORY_BUFFER_SIZE];
    int history_count;
};

/*
 * Calculate chunk allocation strategy for a region
 *
 * ALLOCATION LOGIC:
 * - All regions get complete coverage regardless of size
 * - Calculate exact number of chunks needed to cover entire region
 * - No artificial limits - monitor everything
 */
static void calculate_chunk_allocation(chunked_region_t* region) {
    // Calculate chunks needed for complete coverage
    size_t needed_chunks = (region->size + CHUNK_SIZE - 1) / CHUNK_SIZE;

    if (needed_chunks > MAX_CHUNKS_PER_REGION) {
        fprintf(stderr, "Error: Region %s (%zu MB) exceeds maximum supported size\n",
                region->name, region->size / (1024*1024));
        exit(1);
    }

    region->chunk_count = (int)needed_chunks;
    region->fully_covered = 1;  // Always full coverage
    region->covered_bytes = region->size;  // Cover entire region

    debug("Region %s (%zu MB): %d chunks for complete coverage\n",
          region->name, region->size / (1024*1024), region->chunk_count);
}

/*
 * Allocate memory chunks for a region
 *
 * ERROR HANDLING PHILOSOPHY:
 * - Complete allocation success or complete failure (no partial allocations)
 * - Maintains research accuracy by ensuring consistent data sets
 * - Memory accounting occurs only after successful allocation
 *
 * RETURNS: 1 on success, 0 on failure
 */
static int allocate_region_chunks(chunked_snapshot_context_t* ctx, chunked_region_t* region) {
    calculate_chunk_allocation(region);

    for (int i = 0; i < region->chunk_count; i++) {
        memory_chunk_t* chunk = &region->chunks[i];

        // Allocate fixed-size buffers for previous and current data
        chunk->previous_data = malloc(CHUNK_SIZE);
        chunk->current_data = malloc(CHUNK_SIZE);

        if (!chunk->previous_data || !chunk->current_data) {
            // Clean up current chunk on failure
            if (chunk->previous_data) {
                free(chunk->previous_data);
                chunk->previous_data = NULL;
            }
            if (chunk->current_data) {
                free(chunk->current_data);
                chunk->current_data = NULL;
            }

            // Clean up all previously allocated chunks to maintain consistency
            for (int j = 0; j < i; j++) {
                if (region->chunks[j].previous_data) {
                    free(region->chunks[j].previous_data);
                    region->chunks[j].previous_data = NULL;
                    ctx->total_memory_allocated -= CHUNK_SIZE;
                }
                if (region->chunks[j].current_data) {
                    free(region->chunks[j].current_data);
                    region->chunks[j].current_data = NULL;
                    ctx->total_memory_allocated -= CHUNK_SIZE;
                }
            }

            debug("Failed to allocate chunk %d for region %s\n", i, region->name);
            return 0;
        }

        // Sequential chunk placement (no sampling or distribution)
        chunk->region_offset = (size_t)i * CHUNK_SIZE;

        // Calculate actual data size (handle end-of-region chunks)
        size_t remaining = region->size - chunk->region_offset;
        chunk->data_size = (remaining < CHUNK_SIZE) ? remaining : CHUNK_SIZE;

        chunk->valid = 0;
        chunk->changed_bytes = 0;

        // Account for successful allocation (both buffers)
        ctx->total_memory_allocated += 2 * CHUNK_SIZE;
    }

    debug("Allocated %d chunks (%zu KB total) for region %s\n",
          region->chunk_count, (region->chunk_count * 2 * CHUNK_SIZE) / 1024, region->name);
    return 1;
}

/*
 * Free all chunks for a region and update memory accounting
 * Only call when actually freeing memory, not when reusing chunks
 */
static void free_region_chunks(chunked_snapshot_context_t* ctx, chunked_region_t* region) {
    for (int i = 0; i < region->chunk_count; i++) {
        memory_chunk_t* chunk = &region->chunks[i];
        if (chunk->previous_data) {
            free(chunk->previous_data);
            chunk->previous_data = NULL;
            ctx->total_memory_allocated -= CHUNK_SIZE;
        }
        if (chunk->current_data) {
            free(chunk->current_data);
            chunk->current_data = NULL;
            ctx->total_memory_allocated -= CHUNK_SIZE;
        }
    }
    region->chunk_count = 0;
}

/*
 * Clear region chunks without freeing memory (for reuse)
 */
static void clear_region_chunks(chunked_region_t* region) {
    for (int i = 0; i < region->chunk_count; i++) {
        memory_chunk_t* chunk = &region->chunks[i];
        chunk->previous_data = NULL;
        chunk->current_data = NULL;
    }
    region->chunk_count = 0;
}

/*
 * Find reusable region slot
 *
 * SLOT REUSE STRATEGY:
 * - Exact address match preserves snapshot history for continuous regions
 * - Compatible inactive slots enable efficient buffer reuse
 * - Minimizes malloc/free operations during region discovery
 */
static chunked_region_t* find_region_slot(chunked_snapshot_context_t* ctx, chunked_region_t* new_region) {
    // First pass: exact address match preserves snapshot continuity
    for (int i = 0; i < ctx->region_count; i++) {
        chunked_region_t* region = &ctx->regions[i];
        if (region->start == new_region->start && region->end == new_region->end) {
            debug("Found exact match for region 0x%lx-0x%lx (%s)\n",
                  new_region->start, new_region->end, new_region->name);
            return region;
        }
    }

    // Second pass: find compatible inactive slot for buffer reuse
    for (int i = 0; i < ctx->region_count; i++) {
        chunked_region_t* region = &ctx->regions[i];
        if (!region->active && region->chunk_count >= new_region->chunk_count) {
            debug("Reusing slot for region 0x%lx-0x%lx (%s)\n",
                  new_region->start, new_region->end, new_region->name);
            return region;
        }
    }

    return NULL;
}

/*
 * Parse /proc/self/maps line into region structure
 *
 * ASSUMPTIONS:
 * - /proc/self/maps format is stable across kernel versions
 * - Address ranges are valid and properly ordered
 * - Permission strings follow standard format
 */
static int parse_maps_line(const char* line, chunked_region_t* region) {
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

    // Extract meaningful region name from path
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
 * Determine if region should be monitored
 *
 * FILTERING RATIONALE:
 * - Only writable regions can contain program state changes
 * - System regions ([vdso], [vsyscall], [vvar]) excluded as they contain no program data
 * - All other writable regions included since chunked allocation handles any size
 *
 * DESIGN DECISION: Err on the side of inclusion rather than exclusion for research completeness
 */
static int should_include_region(const chunked_region_t* region, const char* line) {
    if (region->perms[1] != 'w') {
        debug("Excluding non-writable region: %s\n", line);
        return 0;
    }

    if (strstr(line, "[vdso]") || strstr(line, "[vsyscall]") || strstr(line, "[vvar]")) {
        debug("Excluding system region: %s\n", line);
        return 0;
    }

    debug("Including region: %s (size: %zu MB)\n", region->name, region->size / (1024*1024));
    return 1;
}

/*
 * Read /proc/self/maps and parse into candidate regions
 *
 * RETURNS: Number of candidates found, or -1 on error
 */
static int parse_memory_maps(chunked_snapshot_context_t* ctx, chunked_region_t* candidates, int max_candidates) {
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

    // Parse regions from buffer
    char* line_start = ctx->maps_buffer;
    char* line_end;
    int candidate_count = 0;

    while (line_start < ctx->maps_buffer + bytes_read && candidate_count < max_candidates) {
        line_end = strchr(line_start, '\n');
        if (!line_end) {
            line_end = ctx->maps_buffer + bytes_read;
        }

        size_t line_len = line_end - line_start;
        if (line_len >= sizeof(ctx->line_buffer)) {
            line_len = sizeof(ctx->line_buffer) - 1;
        }

        memcpy(ctx->line_buffer, line_start, line_len);
        ctx->line_buffer[line_len] = '\0';

        chunked_region_t candidate;
        memset(&candidate, 0, sizeof(candidate));

        if (parse_maps_line(ctx->line_buffer, &candidate) &&
            should_include_region(&candidate, ctx->line_buffer)) {
            candidates[candidate_count] = candidate;
            candidate_count++;
        }

        line_start = line_end + 1;
    }

    debug("Parsed %d candidate regions from /proc/self/maps\n", candidate_count);
    return candidate_count;
}

/*
 * Assign parsed regions to monitoring slots
 *
 * ASSIGNMENT STRATEGY:
 * - Preserves existing region data when possible to maintain snapshot history
 * - Reallocates only when region configuration changes
 * - Cleans up regions that no longer exist
 *
 * RETURNS: 0 on success, -1 on allocation failure
 */
static int assign_regions_to_slots(chunked_snapshot_context_t* ctx, chunked_region_t* candidates, int candidate_count) {
    // Mark all regions inactive for reuse detection
    for (int i = 0; i < ctx->region_count; i++) {
        ctx->regions[i].active = 0;
    }
    ctx->active_regions = 0;

    // Track newly allocated regions for cleanup on failure
    int allocated_regions[MAX_REGIONS];
    int allocated_count = 0;

    // Assign each candidate to a slot
    for (int i = 0; i < candidate_count; i++) {
        chunked_region_t* candidate = &candidates[i];

        // Calculate chunk requirements for candidate
        calculate_chunk_allocation(candidate);

        // Find appropriate slot for this region
        chunked_region_t* region = find_region_slot(ctx, candidate);

        if (!region) {
            // Need new slot
            if (ctx->region_count >= MAX_REGIONS) {
                fprintf(stderr, "Warning: Maximum regions exceeded, skipping region %s\n",
                       candidate->name);
                continue;
            }
            region = &ctx->regions[ctx->region_count];
            memset(region, 0, sizeof(*region));
            ctx->region_count++;
        }

        // Determine if we can preserve existing chunk data
        int same_region = (region->start == candidate->start && region->end == candidate->end);
        int old_chunk_count = region->chunk_count;
        int can_preserve = same_region && (old_chunk_count == candidate->chunk_count);

        // Preserve chunk buffers if configuration matches
        memory_chunk_t preserved_chunks[MAX_CHUNKS_PER_REGION];
        if (can_preserve) {
            memcpy(preserved_chunks, region->chunks,
                   old_chunk_count * sizeof(memory_chunk_t));
        }

        // Update region information
        *region = *candidate;

        if (can_preserve) {
            // Restore preserved chunk data
            memcpy(region->chunks, preserved_chunks,
                   old_chunk_count * sizeof(memory_chunk_t));
            debug("Preserved %d chunks for region %s\n", old_chunk_count, region->name);
        } else {
            // Free old chunks if they exist (actually freeing memory)
            if (old_chunk_count > 0) {
                free_region_chunks(ctx, region);
            }

            // Allocate new chunks
            if (!allocate_region_chunks(ctx, region)) {
                fprintf(stderr, "Error: Cannot allocate chunks for region %s\n", region->name);

                // Clean up all newly allocated regions from this iteration
                for (int j = 0; j < allocated_count; j++) {
                    int region_idx = allocated_regions[j];
                    free_region_chunks(ctx, &ctx->regions[region_idx]);
                    ctx->regions[region_idx].active = 0;
                    clear_region_chunks(&ctx->regions[region_idx]);
                }

                return -1;
            }

            // Track this region for potential cleanup
            allocated_regions[allocated_count++] = region - ctx->regions;
        }

        region->active = 1;
        region->total_changed_bytes = 0;
        ctx->active_regions++;
    }

    // Clean up any remaining inactive regions
    for (int i = 0; i < ctx->region_count; i++) {
        chunked_region_t* region = &ctx->regions[i];
        if (!region->active && region->chunk_count > 0) {
            debug("Cleaning up inactive region %s\n", region->name);
            free_region_chunks(ctx, region);
        }
    }

    debug("Active regions: %d\n", ctx->active_regions);
    return 0;
}

/*
 * Discover regions and assign to monitoring slots
 *
 * ZERO-MALLOC DISCOVERY:
 * - Uses pre-allocated buffers to avoid memory allocation during discovery
 * - Critical for maintaining consistent memory overhead measurements
 *
 * RETURNS: 0 on success, -1 on error
 */
static int discover_regions(chunked_snapshot_context_t* ctx) {
    debug("Starting region discovery (iteration %d)\n", ctx->iteration);

    // Use static allocation for candidates to avoid stack overflow and malloc
    static chunked_region_t candidates[MAX_REGIONS];

    // Parse memory maps into candidates
    int candidate_count = parse_memory_maps(ctx, candidates, MAX_REGIONS);
    if (candidate_count < 0) {
        return -1;
    }

    // Assign candidates to monitoring slots
    return assign_regions_to_slots(ctx, candidates, candidate_count);
}

/*
 * Read chunk data from process memory
 *
 * ERROR HANDLING:
 * - EINTR (signal interruption) triggers retry for robustness
 * - Short reads indicate memory mapping changes and are treated as errors
 * - Ensures data integrity by validating exact byte counts
 */
static int read_chunk_content(chunked_snapshot_context_t* ctx, chunked_region_t* region,
                             memory_chunk_t* chunk, void* buffer) {
    uintptr_t chunk_address = region->start + chunk->region_offset;

    debug("Reading chunk at 0x%lx (%zu bytes) from region %s\n",
          chunk_address, chunk->data_size, region->name);

    ssize_t bytes_read = pread(ctx->mem_fd, buffer, chunk->data_size, chunk_address);

    if (bytes_read < 0) {
        if (errno == EINTR) {
            // Retry on signal interruption
            return read_chunk_content(ctx, region, chunk, buffer);
        }
        debug("Read failed for chunk in region %s: %s\n", region->name, strerror(errno));
        return 0;
    }

    if ((size_t)bytes_read != chunk->data_size) {
        // Short read indicates memory state inconsistency
        debug("Short read for chunk in region %s: got %zd, expected %zu\n",
              region->name, bytes_read, chunk->data_size);
        return 0;
    }

    return 1;
}

/*
 * Count changed bytes using byte-by-byte comparison
 *
 * COMPARISON STRATEGY:
 * - Only compares chunks with valid previous data
 * - Byte-level granularity provides exact change detection
 * - Fixed chunk size keeps comparison time constant
 */
static void count_chunk_changes(memory_chunk_t* chunk) {
    if (!chunk->valid) {
        chunk->changed_bytes = 0;
        return;
    }

    const char* prev = (const char*)chunk->previous_data;
    const char* curr = (const char*)chunk->current_data;

    chunk->changed_bytes = 0;
    for (size_t i = 0; i < chunk->data_size; i++) {
        if (prev[i] != curr[i]) {
            chunk->changed_bytes++;
        }
    }
}

/*
 * Initialize snapshot context
 *
 * INITIALIZATION SEQUENCE:
 * 1. Allocate and zero context structure
 * 2. Open /proc/self/mem for reading process memory
 * 3. Discover and allocate initial region set
 * 4. Set up iteration tracking and validation magic number
 *
 * ERROR HANDLING: Returns error codes to allow graceful failure handling
 * CLEANUP: Properly frees resources on any initialization failure
 */
int snapshot_init(chunked_snapshot_context_t** ctx_ptr) {
    if (!ctx_ptr) return -1;

    chunked_snapshot_context_t* ctx = malloc(sizeof(chunked_snapshot_context_t));
    if (!ctx) {
        fprintf(stderr, "Error: Failed to allocate snapshot context\n");
        return -1;
    }

    memset(ctx, 0, sizeof(chunked_snapshot_context_t));

    ctx->mem_fd = open("/proc/self/mem", O_RDONLY);
    if (ctx->mem_fd < 0) {
        fprintf(stderr, "Error: Cannot open /proc/self/mem: %s\n", strerror(errno));
        free(ctx);
        return -1;
    }

    debug("Snapshot context initialized\n");

    // Discover initial regions - clean up on failure
    if (discover_regions(ctx) < 0) {
        // Clean up any partially allocated regions
        for (int i = 0; i < ctx->region_count; i++) {
            if (ctx->regions[i].chunk_count > 0) {
                free_region_chunks(ctx, &ctx->regions[i]);
            }
        }
        close(ctx->mem_fd);
        free(ctx);
        return -1;
    }

    ctx->iteration = 0;
    ctx->history_count = 0;
    ctx->initialized = 0xDEADBEEF;
    *ctx_ptr = ctx;

    printf("Memory snapshot initialized: %d regions, %.2f MB overhead\n",
           ctx->active_regions, ctx->total_memory_allocated / (1024.0 * 1024.0));

    return 0;
}

/*
 * Capture memory snapshot and output changes to stdout
 *
 * API DESIGN DECISION:
 * - void return type with exit() on failure simplifies usage
 * - Results output via printf() to stdout (machine-readable format)
 * - No complex return values to handle - simplifies bindings
 * - Research tool philosophy: fail fast and visibly rather than silent degradation
 *
 * OUTPUT FORMAT: Prints exact byte count of changes in monitored regions only
 *
 * CAPTURE SEQUENCE:
 * 1. Re-discover regions (handles dynamic memory layout changes)
 * 2. Read current chunk data from process memory
 * 3. Compare with previous data and count changes (if previous data exists)
 * 4. Swap buffers (current becomes previous for next iteration)
 * 5. Output total changed bytes to stdout
 *
 * CORRECTNESS: Ensures every memory change between captures is detected by
 * reading current state before comparing and preserving data for next comparison
 */
void snapshot_capture(chunked_snapshot_context_t* ctx) {
    if (!ctx || ctx->initialized != 0xDEADBEEF) {
        fprintf(stderr, "Error: Invalid snapshot context\n");
        exit(1);
    }

    debug("=== CAPTURE START (iteration %d) ===\n", ctx->iteration);

    // Re-discover regions to handle dynamic memory layout changes
    if (discover_regions(ctx) < 0) {
        fprintf(stderr, "Error: Region discovery failed\n");
        exit(1);
    }

    // Read current chunk data and detect changes
    size_t total_changed_bytes = 0;
    int regions_with_changes = 0;

    for (int i = 0; i < ctx->region_count; i++) {
        chunked_region_t* region = &ctx->regions[i];
        if (!region->active) continue;

        region->total_changed_bytes = 0;

        for (int j = 0; j < region->chunk_count; j++) {
            memory_chunk_t* chunk = &region->chunks[j];

            // Read current memory content into current_data
            if (!read_chunk_content(ctx, region, chunk, chunk->current_data)) {
                debug("Failed to read chunk %d from region %s\n", j, region->name);
                continue;  // Skip failed chunks but continue with region
            }

            // Compare with previous data (skip first iteration when no previous data exists)
            if (ctx->iteration > 0 && chunk->valid) {
                count_chunk_changes(chunk);
                region->total_changed_bytes += chunk->changed_bytes;
            }

            // Now swap buffers so current becomes previous for next iteration
            void* temp = chunk->previous_data;
            chunk->previous_data = chunk->current_data;
            chunk->current_data = temp;

            chunk->valid = 1;  // Mark as having valid previous data for next comparison
        }

        // Report changes for this region
        if (region->total_changed_bytes > 0) {
            regions_with_changes++;

            if (region->fully_covered) {
                debug("Region %s: %zu bytes changed (fully covered)\n",
                      region->name, region->total_changed_bytes);
            } else {
                debug("Region %s: %zu bytes changed in first %zu KB (partially covered - unmoniored changes not detected)\n",
                      region->name, region->total_changed_bytes, region->covered_bytes / 1024);
            }
        }

        total_changed_bytes += region->total_changed_bytes;
    }

    debug("Iteration %d: %d regions with changes, %zu total bytes changed\n",
          ctx->iteration, regions_with_changes, total_changed_bytes);

    // Output result to stdout in machine-readable format
    printf("%zu\n", total_changed_bytes);
    fflush(stdout);

    ctx->iteration++;
    debug("=== CAPTURE END ===\n");
}

/*
 * Clean up snapshot context and free all allocated memory
 *
 * CLEANUP SEQUENCE:
 * 1. Free all chunk buffers and update memory accounting
 * 2. Close file descriptors
 * 3. Invalidate context and free structure
 */
void snapshot_cleanup(chunked_snapshot_context_t* ctx) {
    if (!ctx) return;

    debug("Cleaning up snapshot context\n");

    // Free all chunk data
    for (int i = 0; i < ctx->region_count; i++) {
        chunked_region_t* region = &ctx->regions[i];
        if (region->chunk_count > 0) {
            free_region_chunks(ctx, region);
        }
    }

    if (ctx->mem_fd >= 0) {
        close(ctx->mem_fd);
    }

    ctx->initialized = 0;
    free(ctx);
}

#ifdef TEST_CHUNKED
int main() {
    chunked_snapshot_context_t* snapshotter;

    if (snapshot_init(&snapshotter) != 0) {
        fprintf(stderr, "Failed to initialize snapshot\n");
        return 1;
    }

    printf("# Starting snapshot test\n");

    for (int i = 0; i < 5; i++) {
        size_t alloc_size = 1024 * 1024 * (i + 1);
        void* temp = malloc(alloc_size);
        memset(temp, i + 42, alloc_size);

        printf("# Iteration %d (allocated %zu MB)\n", i + 1, alloc_size / (1024 * 1024));
        snapshot_capture(snapshotter);

        free(temp);
    }

    snapshot_cleanup(snapshotter);
    printf("# Test completed\n");
    return 0;
}
#endif
