#include "event_parser.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

namespace replay {

namespace {

// Minimal JSON string value extractor — avoids external dependency.
// Handles simple flat objects with string/int values.
std::string extract_string(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";

    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return "";
    ++pos;

    // Skip whitespace
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;

    if (pos < json.size() && json[pos] == '"') {
        auto start = pos + 1;
        auto end = json.find('"', start);
        if (end == std::string::npos) return "";
        return json.substr(start, end - start);
    }
    return "";
}

int64_t extract_int64(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return 0;

    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return 0;
    ++pos;

    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;

    std::string num_str;
    while (pos < json.size() && (std::isdigit(json[pos]) || json[pos] == '-')) {
        num_str += json[pos++];
    }
    if (num_str.empty()) return 0;
    return std::stoll(num_str);
}

int extract_int(const std::string& json, const std::string& key) {
    return static_cast<int>(extract_int64(json, key));
}

}  // namespace

Event parse_event_line(const std::string& line) {
    Event evt;
    evt.timestamp_ns = extract_int64(line, "timestamp_ns");
    evt.request_id = extract_string(line, "request_id");
    evt.service_name = extract_string(line, "service_name");
    evt.phase = extract_string(line, "phase");
    evt.event_type = extract_string(line, "event_type");
    evt.sequence_no = extract_int(line, "sequence_no");
    evt.payload_hash = extract_string(line, "payload_hash");
    return evt;
}

std::vector<Event> parse_event_log(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open log file: " + filepath);
    }

    std::vector<Event> events;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] != '{') continue;
        events.push_back(parse_event_line(line));
    }
    return events;
}

}  // namespace replay
