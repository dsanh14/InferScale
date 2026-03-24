#pragma once

#include <string>
#include <vector>
#include <unordered_map>

namespace replay {

struct Event {
    int64_t timestamp_ns = 0;
    std::string request_id;
    std::string service_name;
    std::string phase;
    std::string event_type;
    int sequence_no = 0;
    std::string payload_hash;
    std::unordered_map<std::string, std::string> metadata;
};

/// Parse a JSONL file into a vector of Events.
/// Each line is a JSON object matching the Python Event schema.
std::vector<Event> parse_event_log(const std::string& filepath);

/// Parse a single JSON line into an Event.
Event parse_event_line(const std::string& line);

}  // namespace replay
