#pragma once

#include "event_parser.h"
#include <string>
#include <vector>

namespace replay {

/// SHA-256 hash of the stable (deterministic) fields of a single event.
/// Mirrors the Python hashing.hash_event() logic.
std::string hash_event(const Event& event);

/// Cumulative hash over an ordered trace of events.
/// Each event hash is folded into a running SHA-256 state.
std::string hash_trace(const std::vector<Event>& events);

/// Raw SHA-256 of an arbitrary string, returned as hex.
std::string sha256_hex(const std::string& input);

}  // namespace replay
