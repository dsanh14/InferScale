#include "replay_validator.h"
#include "state_hasher.h"

#include <iostream>
#include <sstream>

namespace replay {

ValidationReport validate(
    const std::vector<Event>& trace_a,
    const std::vector<Event>& trace_b
) {
    ValidationReport report;
    report.total_events_a = trace_a.size();
    report.total_events_b = trace_b.size();

    report.trace_hash_a = hash_trace(trace_a);
    report.trace_hash_b = hash_trace(trace_b);

    if (report.trace_hash_a == report.trace_hash_b) {
        report.match = true;
        report.summary = "Traces are deterministically equivalent.";
        return report;
    }

    report.match = false;

    // Find the first divergence point
    size_t min_len = std::min(trace_a.size(), trace_b.size());
    for (size_t i = 0; i < min_len; ++i) {
        std::string ha = hash_event(trace_a[i]);
        std::string hb = hash_event(trace_b[i]);
        if (ha != hb) {
            report.first_divergence_index = static_cast<int>(i);
            break;
        }
    }

    if (report.first_divergence_index < 0 && trace_a.size() != trace_b.size()) {
        report.first_divergence_index = static_cast<int>(min_len);
    }

    std::ostringstream oss;
    oss << "DIVERGENCE DETECTED at event index " << report.first_divergence_index << ".\n";
    if (report.first_divergence_index >= 0 &&
        static_cast<size_t>(report.first_divergence_index) < min_len) {
        const auto& ea = trace_a[report.first_divergence_index];
        const auto& eb = trace_b[report.first_divergence_index];
        oss << "  Trace A: service=" << ea.service_name
            << " phase=" << ea.phase
            << " type=" << ea.event_type
            << " seq=" << ea.sequence_no << "\n";
        oss << "  Trace B: service=" << eb.service_name
            << " phase=" << eb.phase
            << " type=" << eb.event_type
            << " seq=" << eb.sequence_no << "\n";
    }
    report.summary = oss.str();
    return report;
}

void print_report(const ValidationReport& report) {
    std::cout << "--- Validation Report ---\n";
    std::cout << "Events in A: " << report.total_events_a << "\n";
    std::cout << "Events in B: " << report.total_events_b << "\n";
    std::cout << "Trace hash A: " << report.trace_hash_a << "\n";
    std::cout << "Trace hash B: " << report.trace_hash_b << "\n";
    std::cout << "Match: " << (report.match ? "YES" : "NO") << "\n";
    if (!report.match) {
        std::cout << "First divergence at index: " << report.first_divergence_index << "\n";
    }
    std::cout << "\n" << report.summary << "\n";
}

}  // namespace replay
