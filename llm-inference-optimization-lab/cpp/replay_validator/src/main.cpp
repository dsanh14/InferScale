#include "event_parser.h"
#include "replay_validator.h"

#include <cstdlib>
#include <iostream>
#include <string>

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --log_a <path> --log_b <path>\n\n"
              << "Compare two structured event logs for deterministic equivalence.\n"
              << "Exit code 0 = match, 1 = divergence detected.\n";
}

int main(int argc, char* argv[]) {
    std::string path_a, path_b;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--log_a" && i + 1 < argc) {
            path_a = argv[++i];
        } else if (arg == "--log_b" && i + 1 < argc) {
            path_b = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (path_a.empty() || path_b.empty()) {
        print_usage(argv[0]);
        return 2;
    }

    std::cout << "=== Deterministic Replay Validator ===\n\n";
    std::cout << "Log A: " << path_a << "\n";
    std::cout << "Log B: " << path_b << "\n\n";

    auto trace_a = replay::parse_event_log(path_a);
    auto trace_b = replay::parse_event_log(path_b);

    std::cout << "Parsed " << trace_a.size() << " events from A, "
              << trace_b.size() << " events from B\n\n";

    auto report = replay::validate(trace_a, trace_b);
    replay::print_report(report);

    return report.match ? 0 : 1;
}
