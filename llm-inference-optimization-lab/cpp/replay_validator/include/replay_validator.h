#pragma once

#include "event_parser.h"
#include <string>
#include <vector>

namespace replay {

struct ValidationReport {
    bool match = true;
    size_t total_events_a = 0;
    size_t total_events_b = 0;
    int first_divergence_index = -1;
    std::string trace_hash_a;
    std::string trace_hash_b;
    std::string summary;
};

/// Compare two event traces for deterministic equivalence.
/// Returns a report describing whether the traces match and
/// where the first divergence occurs if they differ.
ValidationReport validate(
    const std::vector<Event>& trace_a,
    const std::vector<Event>& trace_b
);

/// Pretty-print the validation report to stdout.
void print_report(const ValidationReport& report);

}  // namespace replay
