#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;

const int NUM_PARTS = 10;
const int RUN_ID_MIN = 2201000000;
const int RUN_ID_MAX = 2201000999;
const int BIN_SIZE = (RUN_ID_MAX - RUN_ID_MIN + 1) / NUM_PARTS;

// Trim leading/trailing whitespace and remove non-numeric characters
std::string clean_number(const std::string &str) {
    std::string result;
    for (char c : str) {
        if (std::isdigit(c)) {  // Keep only numeric characters
            result += c;
        }
    }
    return result;
}

// Function to determine which file a row should go to
int get_part_index(int run_id) {
    return (run_id - RUN_ID_MIN) / BIN_SIZE;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " input.csv\n";
        return 1;
    }

    std::string input_file = argv[1];
    std::ifstream infile(input_file);
    if (!infile.is_open()) {
        std::cerr << "Error opening input file: " << input_file << "!\n";
        return 1;
    }

    // Extract parent directory from input file
    fs::path input_path(input_file);
    fs::path output_dir = input_path.parent_path();

    // Read and verify the header
    std::string header;
    if (!std::getline(infile, header)) {
        std::cerr << "Error: Empty file or missing header in " << input_file << "!\n";
        return 1;
    }

    // Open multiple output files with correct naming
    std::vector<std::ofstream> outfiles(NUM_PARTS);
    for (int i = 0; i < NUM_PARTS; i++) {
        int start_range = RUN_ID_MIN + i * BIN_SIZE;
        int end_range = start_range + BIN_SIZE - 1;

        std::string output_filename = "clean_event_ids_" +
                                      std::to_string(start_range - RUN_ID_MIN).insert(0, 7 - std::to_string(start_range - RUN_ID_MIN).length(), '0') + "-" +
                                      std::to_string(end_range - RUN_ID_MIN).insert(0, 7 - std::to_string(end_range - RUN_ID_MIN).length(), '0') + ".csv";

        fs::path output_file = output_dir / output_filename;
        outfiles[i].open(output_file);
        if (!outfiles[i].is_open()) {
            std::cerr << "Error opening output file: " << output_file << "\n";
            return 1;
        }
        outfiles[i] << header << "\n";  // Write header to each file
    }

    std::string line;
    while (std::getline(infile, line)) {
        std::stringstream ss(line);
        std::string run_id_str, event_id_str;

        if (!std::getline(ss, run_id_str, ',') || !std::getline(ss, event_id_str, ',')) {
            continue;  // Skip malformed lines
        }

        // Clean numbers to remove spaces, \r, and other invalid characters
        run_id_str = clean_number(run_id_str);
        event_id_str = clean_number(event_id_str);

        try {
            int run_id = std::stoi(run_id_str);
            int event_id = std::stoi(event_id_str);  // Ensure it's a valid integer
            int part_index = get_part_index(run_id);

            if (part_index >= 0 && part_index < NUM_PARTS) {
                outfiles[part_index] << run_id << "," << event_id << "\n";  // Write to correct file
            } else {
                std::cerr << "Skipping row with out-of-range RunID: " << run_id << "\n";
            }
        } catch (const std::exception &e) {
            std::cerr << "Skipping invalid row: " << line << " (Error: " << e.what() << ")\n";
        }
    }

    // Close all files
    infile.close();
    for (auto &outfile : outfiles) {
        outfile.close();
    }

    std::cout << "Splitting completed! Files saved in: " << output_dir << "\n";
    return 0;
}
