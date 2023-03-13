#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <algorithm>

static std::vector<char> read_file(const char *filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Error opening " << filename << '\n';
        exit(1);
    }
    std::size_t size = file.tellg();
    std::vector<char> buffer(size);

    file.seekg(0, std::ios::beg);
    file.read(buffer.data(), size);

    return buffer;
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << "i file1 file2 chunk\n";
        return 1;
    }
    const int i = std::stoi(argv[1]);
    const std::vector<char> buffer1 = read_file(argv[2]);
    const std::vector<char> buffer2 = read_file(argv[3]);
    if (buffer1.size() != buffer2.size()) {
        std::cerr << "Error: file sizes differ\n";
        return 1;
    }
    std::size_t size = buffer1.size();

    const int chunk_size = std::stoi(argv[4]);

    int count = 0;

    for (std::size_t i = 0; i < size; i += chunk_size) {
        std::size_t chunk_end = std::min(i + chunk_size, size);

        if (std::memcmp(buffer1.data() + i, buffer2.data() + i, chunk_end - i) != 0) {
            count++;
        }
    }

    std::cout << i << "," << count * chunk_size  << "\n";  // << " bytes (" << count << " chunks of size " << chunk_size << ")\n";

    return 0;
}
