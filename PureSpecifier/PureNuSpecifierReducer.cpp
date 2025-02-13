#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_set>

using namespace std;

struct RowHash {
    size_t operator()(const string &s) const {
        return hash<string>{}(s);
    }
};

int main(int argc, char *argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " input.csv output.csv\n";
        return 1;
    }

    ifstream infile(argv[1]);
    ofstream outfile(argv[2]);

    if (!infile.is_open() || !outfile.is_open()) {
        cerr << "Error opening files!\n";
        return 1;
    }

    string line;
    unordered_set<string, RowHash> unique_rows;

    // Read header and write the new reduced header to output
    if (getline(infile, line)) {
        outfile << "Run ID,Event ID,\n";
    }

    while (getline(infile, line)) {
        stringstream ss(line);
        string run_id, subrun_id, event_id, subevent_id, subevent_stream;

        // Read CSV columns
        if (!getline(ss, run_id, ',') ||
            !getline(ss, subrun_id, ',') ||
            !getline(ss, event_id, ',') ||
            !getline(ss, subevent_id, ',') ||
            !getline(ss, subevent_stream, ',')) {
            continue;  // Skip malformed lines
        }

        string row_key = run_id + "," + event_id + ",";  // Keep only Run ID and Event ID

        if (unique_rows.insert(row_key).second) {
            outfile << row_key << "\n";  // Write reduced row
        }
    }

    infile.close();
    outfile.close();
    return 0;
}
