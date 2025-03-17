#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_set>

struct RowHash {
    size_t operator()(const std::string &s) const {
        return std::hash<std::string>()(s);
    }
};

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " input.csv output.csv\n";
        return 1;
    }

    std::ifstream infile(argv[1]);
    std::ofstream outfile(argv[2]);

    if (!infile.is_open() || !outfile.is_open()) {
        std::cerr << "Error opening files!\n";
        return 1;
    }

    std::string line;
    std::unordered_set<std::string, RowHash> unique_rows;

    // Read and modify header
    if (std::getline(infile, line)) {
        outfile << "RunID,EventID\n";  // Standardized header
    }

    while (std::getline(infile, line)) {
        std::stringstream ss(line);
        std::string run_id, subrun_id, event_id, subevent_id, subevent_stream;

        // Read known columns (Subevent Stream may be missing)
        if (!std::getline(ss, run_id, ',') ||
            !std::getline(ss, subrun_id, ',') ||
            !std::getline(ss, event_id, ',') ||
            !std::getline(ss, subevent_id, ',')) {
            continue;  // Skip malformed lines
        }

        // Read subevent_stream (optional)
        std::getline(ss, subevent_stream, ',');

        std::string row_key = run_id + "," + event_id;  // Keep only RunID and EventID

        if (unique_rows.insert(row_key).second) {
            outfile << row_key << "\n";  // Write reduced row without trailing comma
        }
    }

    infile.close();
    outfile.close();
    return 0;
}
