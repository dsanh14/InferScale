#include "event_parser.h"
#include "state_hasher.h"
#include "replay_validator.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

namespace {

void write_test_log(const std::string& path, const std::vector<std::string>& lines) {
    std::ofstream f(path);
    for (const auto& line : lines) f << line << "\n";
}

std::string make_event_json(
    const std::string& req_id, const std::string& service,
    const std::string& phase, const std::string& event_type,
    int seq, const std::string& payload_hash
) {
    return "{\"timestamp_ns\": 1000000000, \"request_id\": \"" + req_id +
           "\", \"service_name\": \"" + service +
           "\", \"phase\": \"" + phase +
           "\", \"event_type\": \"" + event_type +
           "\", \"sequence_no\": " + std::to_string(seq) +
           ", \"payload_hash\": \"" + payload_hash +
           "\", \"metadata\": {}}";
}

void test_parse_event_line() {
    std::string line = make_event_json("abc123", "baseline", "inference", "generate_start", 0, "deadbeef");
    auto evt = replay::parse_event_line(line);
    assert(evt.request_id == "abc123");
    assert(evt.service_name == "baseline");
    assert(evt.phase == "inference");
    assert(evt.event_type == "generate_start");
    assert(evt.sequence_no == 0);
    assert(evt.payload_hash == "deadbeef");
    std::cout << "  PASS: test_parse_event_line\n";
}

void test_hash_determinism() {
    replay::Event a, b;
    a.request_id = b.request_id = "req1";
    a.service_name = b.service_name = "svc";
    a.phase = b.phase = "p";
    a.event_type = b.event_type = "e";
    a.sequence_no = b.sequence_no = 1;
    a.payload_hash = b.payload_hash = "hash1";

    assert(replay::hash_event(a) == replay::hash_event(b));
    std::cout << "  PASS: test_hash_determinism\n";
}

void test_matching_traces() {
    std::vector<std::string> lines = {
        make_event_json("r1", "svc", "p1", "start", 0, "h1"),
        make_event_json("r1", "svc", "p1", "end", 1, "h2"),
    };

    std::string path_a = "/tmp/test_trace_a.jsonl";
    std::string path_b = "/tmp/test_trace_b.jsonl";
    write_test_log(path_a, lines);
    write_test_log(path_b, lines);

    auto ta = replay::parse_event_log(path_a);
    auto tb = replay::parse_event_log(path_b);

    auto report = replay::validate(ta, tb);
    assert(report.match);
    assert(report.first_divergence_index == -1);
    std::cout << "  PASS: test_matching_traces\n";

    std::filesystem::remove(path_a);
    std::filesystem::remove(path_b);
}

void test_divergent_traces() {
    std::vector<std::string> lines_a = {
        make_event_json("r1", "svc", "p1", "start", 0, "h1"),
        make_event_json("r1", "svc", "p1", "end", 1, "h2"),
    };
    std::vector<std::string> lines_b = {
        make_event_json("r1", "svc", "p1", "start", 0, "h1"),
        make_event_json("r1", "svc", "p1", "end", 1, "DIFFERENT"),
    };

    std::string path_a = "/tmp/test_div_a.jsonl";
    std::string path_b = "/tmp/test_div_b.jsonl";
    write_test_log(path_a, lines_a);
    write_test_log(path_b, lines_b);

    auto ta = replay::parse_event_log(path_a);
    auto tb = replay::parse_event_log(path_b);

    auto report = replay::validate(ta, tb);
    assert(!report.match);
    assert(report.first_divergence_index == 1);
    std::cout << "  PASS: test_divergent_traces\n";

    std::filesystem::remove(path_a);
    std::filesystem::remove(path_b);
}

}  // namespace

int main() {
    std::cout << "Running replay validator tests...\n";
    test_parse_event_line();
    test_hash_determinism();
    test_matching_traces();
    test_divergent_traces();
    std::cout << "\nAll tests passed.\n";
    return 0;
}
