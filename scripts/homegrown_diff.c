/*
 * Memory Snapshot Differ
 *
 * PURPOSE:
 * Compares memory snapshots and creates diff size information.
 * Designed to work with the snapshot tool's output directories.
 *
 * OPERATION:
 * 1. Takes a snapshot directory N/ as input
 * 2. Follows the 'parent' symlink to find the previous snapshot N-1/
 * 3. Compares corresponding region_*.bin files byte-by-byte
 * 4. Creates region_*_diffsize.txt files in the CHILD directory N/
 * 5. Creates summary diffsize.txt in the CHILD directory N/
 *
 * DIRECTORY STRUCTURE:
 * Before:
 *   N-1/region_abc.bin, N-1/regions.txt
 *   N/region_abc.bin, N/parent -> ../N-1/, N/regions.txt
 *
 * After:
 *   N-1/region_abc.bin, N-1/regions.txt  (unchanged)
 *   N/region_abc.bin, N/parent -> ../N-1/, N/regions.txt
 *   N/region_abc_diffsize.txt (contains byte diff count)
 *   N/diffsize.txt (summary of all diffs)
 *
 * USAGE:
 *   ./snapshot_differ snapshots/5/
 *   # Compares snapshots/5/ with snapshots/4/ (via parent link)
 *   # Creates diff files in snapshots/5/
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>
#include <stdarg.h>

#define READ_BUFFER_SIZE (1024 * 1024)  // 1MB buffer for file comparison

// Debug control via environment variable
static int debug_enabled = -1;

static void debug(const char* fmt, ...) {
    if (debug_enabled == -1) {
        debug_enabled = getenv("DIFFER_DEBUG") ? 1 : 0;
    }
    if (!debug_enabled) return;

    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "DEBUG: ");
    vfprintf(stderr, fmt, args);
    va_end(args);
}

/*
 * Resolve parent directory path by following the 'parent' symlink
 *
 * INPUT: snapshot_dir (e.g., "snapshots/5/")
 * OUTPUT: parent_dir (e.g., "snapshots/4/")
 * RETURNS: 0 on success, -1 if no parent link exists
 */
static int resolve_parent_dir(const char* snapshot_dir, char* parent_dir, size_t parent_size) {
    char parent_link[512];
    snprintf(parent_link, sizeof(parent_link), "%s/parent", snapshot_dir);

    // Read the symlink target
    ssize_t link_len = readlink(parent_link, parent_dir, parent_size - 1);
    if (link_len == -1) {
        if (errno == ENOENT) {
            debug("No parent link found - this is the first snapshot\n");
            return -1;
        }
        fprintf(stderr, "Error: Cannot read parent link %s: %s\n", parent_link, strerror(errno));
        exit(1);
    }

    parent_dir[link_len] = '\0';

    // Convert relative path to absolute if needed
    if (parent_dir[0] != '/') {
        char abs_parent[512];
        snprintf(abs_parent, sizeof(abs_parent), "%s/%s", snapshot_dir, parent_dir);
        strncpy(parent_dir, abs_parent, parent_size - 1);
        parent_dir[parent_size - 1] = '\0';
    }

    debug("Parent directory: %s\n", parent_dir);
    return 0;
}

/*
 * Compare two files byte-by-byte and return number of differing bytes
 *
 * ALGORITHM:
 * - Read both files in chunks
 * - Compare chunk contents byte-by-byte
 * - Count total differing bytes
 *
 * RETURNS: Number of differing bytes, or -1 on error
 */
static long compare_files(const char* file1, const char* file2) {
    int fd1 = open(file1, O_RDONLY);
    int fd2 = open(file2, O_RDONLY);

    if (fd1 < 0) {
        debug("Cannot open %s: %s\n", file1, strerror(errno));
        if (fd2 >= 0) close(fd2);
        return -1;
    }

    if (fd2 < 0) {
        debug("Cannot open %s: %s\n", file2, strerror(errno));
        close(fd1);
        return -1;
    }

    // Get file sizes
    struct stat st1, st2;
    if (fstat(fd1, &st1) != 0 || fstat(fd2, &st2) != 0) {
        fprintf(stderr, "Error: Cannot stat files for comparison\n");
        close(fd1);
        close(fd2);
        return -1;
    }

    char buffer1[READ_BUFFER_SIZE];
    char buffer2[READ_BUFFER_SIZE];
    long diff_bytes = 0;

    debug("Comparing files: %s (%ld bytes) vs %s (%ld bytes)\n",
          file1, st1.st_size, file2, st2.st_size);

    // If sizes differ, the extra bytes count as differences
    if (st1.st_size != st2.st_size) {
        diff_bytes += labs(st1.st_size - st2.st_size);
        debug("Size difference: %ld bytes\n", labs(st1.st_size - st2.st_size));
    }

    // Compare common portion byte-by-byte
    long min_size = (st1.st_size < st2.st_size) ? st1.st_size : st2.st_size;
    long bytes_compared = 0;

    while (bytes_compared < min_size) {
        long chunk_size = (min_size - bytes_compared < READ_BUFFER_SIZE) ?
                         (min_size - bytes_compared) : READ_BUFFER_SIZE;

        ssize_t read1 = read(fd1, buffer1, chunk_size);
        ssize_t read2 = read(fd2, buffer2, chunk_size);

        if (read1 != read2 || read1 <= 0) {
            if (read1 <= 0 || read2 <= 0) break;
            fprintf(stderr, "Error: Read size mismatch during comparison\n");
            break;
        }

        // Compare bytes in this chunk
        for (ssize_t i = 0; i < read1; i++) {
            if (buffer1[i] != buffer2[i]) {
                diff_bytes++;
            }
        }

        bytes_compared += read1;
    }

    close(fd1);
    close(fd2);

    debug("Total differing bytes: %ld\n", diff_bytes);
    return diff_bytes;
}

/*
 * Extract region identifier from filename
 *
 * INPUT: "region_55ecaecc4000.bin"
 * OUTPUT: "55ecaecc4000"
 * RETURNS: 1 if extracted successfully, 0 if not a region file
 */
static int extract_region_id(const char* filename, char* region_id, size_t id_size) {
    if (strncmp(filename, "region_", 7) != 0) {
        return 0;
    }

    const char* start = filename + 7;  // Skip "region_"
    const char* end = strstr(start, ".bin");
    if (!end) {
        return 0;
    }

    size_t id_len = end - start;
    if (id_len >= id_size) {
        return 0;
    }

    strncpy(region_id, start, id_len);
    region_id[id_len] = '\0';
    return 1;
}

/*
 * Write diff size to text file in child directory
 *
 * Creates: snapshot_dir/region_<id>_diffsize.txt
 * Content: Single line with number of differing bytes
 */
static void write_diff_size(const char* snapshot_dir, const char* region_id, long diff_bytes) {
    char diff_file[512];
    snprintf(diff_file, sizeof(diff_file), "%s/region_%s_diffsize.txt", snapshot_dir, region_id);

    FILE* f = fopen(diff_file, "w");
    if (!f) {
        fprintf(stderr, "Error: Cannot create diff file %s: %s\n", diff_file, strerror(errno));
        exit(1);
    }

    fprintf(f, "%ld\n", diff_bytes);
    fclose(f);

    debug("Wrote diff size %ld to %s\n", diff_bytes, diff_file);
}

/*
 * Write summary diff file in child directory
 *
 * Creates: snapshot_dir/diffsize.txt
 * Content: Total number of differing bytes across all regions
 */
static void write_summary_diff(const char* snapshot_dir, long total_diff_bytes) {
    char summary_file[512];
    snprintf(summary_file, sizeof(summary_file), "%s/diffsize.txt", snapshot_dir);

    FILE* f = fopen(summary_file, "w");
    if (!f) {
        fprintf(stderr, "Error: Cannot create summary file %s: %s\n", summary_file, strerror(errno));
        exit(1);
    }

    fprintf(f, "%ld\n", total_diff_bytes);
    fclose(f);

    debug("Wrote summary diff size %ld to %s\n", total_diff_bytes, summary_file);
}

/*
 * Process all region files in the snapshot directory
 *
 * ALGORITHM:
 * 1. Scan current snapshot directory for region_*.bin files
 * 2. For each region file, find corresponding file in parent directory
 * 3. Compare files and calculate diff size
 * 4. Write diff size to child directory (preserves parent data)
 *
 * RETURNS: Total number of differing bytes across all regions
 */
static long process_regions(const char* snapshot_dir, const char* parent_dir) {
    DIR* dir = opendir(snapshot_dir);
    if (!dir) {
        fprintf(stderr, "Error: Cannot open directory %s: %s\n", snapshot_dir, strerror(errno));
        exit(1);
    }

    int regions_processed = 0;
    long total_diff_bytes = 0;
    struct dirent* entry;

    while ((entry = readdir(dir)) != NULL) {
        char region_id[64];
        if (!extract_region_id(entry->d_name, region_id, sizeof(region_id))) {
            continue;  // Skip non-region files
        }

        // Build file paths
        char current_file[512], parent_file[512];
        snprintf(current_file, sizeof(current_file), "%s/%s", snapshot_dir, entry->d_name);
        snprintf(parent_file, sizeof(parent_file), "%s/%s", parent_dir, entry->d_name);

        // Compare files
        long diff_bytes = compare_files(parent_file, current_file);
        if (diff_bytes >= 0) {
            write_diff_size(snapshot_dir, region_id, diff_bytes);
            total_diff_bytes += diff_bytes;
            regions_processed++;
        } else {
            debug("Skipping region %s (comparison failed)\n", region_id);
        }
    }

    closedir(dir);

    debug("Processed %d regions, total diff: %ld bytes\n", regions_processed, total_diff_bytes);

    if (regions_processed == 0) {
        fprintf(stderr, "Warning: No region files found to process\n");
    }

    return total_diff_bytes;
}

/*
 * Main differ function
 *
 * INPUT: Path to current snapshot directory (e.g., "snapshots/5/")
 * OPERATION: Compare with parent and create diff files in child directory
 */
void differ_process_snapshot(const char* snapshot_dir) {
    debug("=== Processing snapshot: %s ===\n", snapshot_dir);

    // Resolve parent directory
    char parent_dir[512];
    if (resolve_parent_dir(snapshot_dir, parent_dir, sizeof(parent_dir)) != 0) {
        printf("No parent found - skipping first snapshot\n");
        return;
    }

    // Process all region files and get total diff
    long total_diff_bytes = process_regions(snapshot_dir, parent_dir);

    // Write summary diff file
    write_summary_diff(snapshot_dir, total_diff_bytes);

    printf("Processed %s vs %s: %ld total differing bytes\n",
           snapshot_dir, parent_dir, total_diff_bytes);
    debug("Created diffsize.txt with total: %ld bytes\n", total_diff_bytes);
    debug("=== Processing complete ===\n");
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <snapshot_directory>\n", argv[0]);
        fprintf(stderr, "Example: %s snapshots/5/\n", argv[0]);
        return 1;
    }

    const char* snapshot_dir = argv[1];

    // Validate directory exists
    struct stat st;
    if (stat(snapshot_dir, &st) != 0 || !S_ISDIR(st.st_mode)) {
        fprintf(stderr, "Error: %s is not a valid directory\n", snapshot_dir);
        return 1;
    }

    differ_process_snapshot(snapshot_dir);
    return 0;
}
