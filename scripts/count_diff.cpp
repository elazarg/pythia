#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <algorithm>
#include <filesystem>
namespace fs = std::filesystem;

static std::vector<char> read_file(const fs::path& filename, bool remove) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Error opening " << filename << '\n';
        exit(1);
    }
    if (remove) fs::remove(filename);

    std::size_t size = file.tellg();
    std::vector<char> buffer(size);

    file.seekg(0, std::ios::beg);
    file.read(buffer.data(), size);

    return buffer;
}

int main(int argc, char *argv[]) {
    if (argc != 4 && argc != 5) {
        std::cerr << "Usage: " << argv[0] << "file1 file2 chunk [remove]\n";
        return 1;
    }
    bool remove = false;
    if (argc == 5) {
        if (std::string(argv[4]) != "remove") {
            std::cerr << "Error: unknown argument " << argv[4] << '\n';
            return 1;
        }
        remove = true;
    }
    const std::vector<char> buffer1 = read_file(argv[1], remove);
    const std::vector<char> buffer2 = read_file(argv[2], remove);

    if (buffer1.size() != buffer2.size()) {
        std::cerr << "Error: file sizes differ\n";
        return 1;
    }
    std::size_t size = buffer1.size();

    const int chunk_size = std::stoi(argv[3]);

    int count = 0;

    for (std::size_t i = 0; i < size; i += chunk_size) {
        std::size_t chunk_end = std::min(i + chunk_size, size);

        if (std::memcmp(buffer1.data() + i, buffer2.data() + i, chunk_end - i) != 0) {
            count++;
        }
    }

    std::cout << count * chunk_size  << "\n";
    return 0;
}
x