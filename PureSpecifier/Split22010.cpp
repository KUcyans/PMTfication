#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <unordered_map>
#include <vector>

// Function to get current timestamp as a string
std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    char buffer[100];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&now_c));
    return std::string(buffer);
}

void splitCSVByRunID(const std::string &inputFile, const std::string &outputDir) {
    std::ifstream infile(inputFile);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open input file: " << inputFile << std::endl;
        return;
    }

    std::unordered_map<long long, std::ofstream> fileMap;
    std::string line;
    std::set<long long> seenRunIDs;  // Track unique RunIDs
    std::set<long long> missingRunIDs = {2201000020LL, 2201000107LL, 2201000731LL, 2201000858LL, 2201000904LL, 2201000963LL};

    std::cout << "[" << getCurrentTimestamp() << "] Started processing file: " << inputFile << std::endl;

    // Read header line and store it for writing to each file
    if (!std::getline(infile, line)) {
        std::cerr << "Error: Input file is empty or not formatted correctly." << std::endl;
        return;
    }
    std::string header = line;

    while (std::getline(infile, line)) {
        std::stringstream ss(line);
        std::string runIDStr, eventIDStr;

        if (!std::getline(ss, runIDStr, ',') || !std::getline(ss, eventIDStr, ',')) {
            std::cerr << "Warning: Skipping malformed line: " << line << std::endl;
            continue;
        }

        // Convert to long long using std::stoll()
        long long runID = std::stoll(runIDStr);
        seenRunIDs.insert(runID);  // Track seen RunIDs

        // Skip known missing RunIDs
        if (missingRunIDs.find(runID) != missingRunIDs.end()) {
            std::cerr << "Skipping missing RunID: " << runID << std::endl;
            continue;
        }

        long long bucket = (runID % 1000) / 100 * 100;  // Extract range (000-099, 100-199, etc.)

        std::string filename = outputDir + "/clean_event_ids_" +
                               std::to_string(bucket).insert(0, 7 - std::to_string(bucket).length(), '0') +
                               "-" +
                               std::to_string(bucket + 99).insert(0, 7 - std::to_string(bucket + 99).length(), '0') +
                               ".csv";

        if (fileMap.find(bucket) == fileMap.end()) {
            fileMap[bucket].open(filename);
            if (!fileMap[bucket].is_open()) {
                std::cerr << "Error: Could not open output file " << filename << std::endl;
                continue;
            }
            fileMap[bucket] << header << std::endl;
        }

        fileMap[bucket] << line << std::endl;
    }

    for (auto &pair : fileMap) {
        pair.second.close();
    }
    infile.close();

    std::cout << "[" << getCurrentTimestamp() << "] Splitting completed successfully! Output saved in: " << outputDir << std::endl;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_directory>" << std::endl;
        return 1;
    }

    std::string inputFile = argv[1];
    std::string outputDir = argv[2];

    // Start time
    auto start = std::chrono::high_resolution_clock::now();

    splitCSVByRunID(inputFile, outputDir);

    // End time
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "[" << getCurrentTimestamp() << "] Execution Time: "
              << elapsed.count() << " seconds (" << elapsed.count() * 1000 << " ms)" << std::endl;

    return 0;
}
