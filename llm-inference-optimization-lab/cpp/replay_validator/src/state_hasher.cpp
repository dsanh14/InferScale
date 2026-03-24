#include "state_hasher.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>

namespace replay {

namespace {

// Minimal SHA-256 implementation (no external dependency).
// Adapted from public domain reference implementations.

static constexpr std::array<uint32_t, 64> K = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
inline uint32_t sig0(uint32_t x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
inline uint32_t sig1(uint32_t x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
inline uint32_t gam0(uint32_t x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
inline uint32_t gam1(uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

std::array<uint8_t, 32> sha256_bytes(const uint8_t* data, size_t len) {
    uint32_t h0 = 0x6a09e667, h1 = 0xbb67ae85, h2 = 0x3c6ef372, h3 = 0xa54ff53a;
    uint32_t h4 = 0x510e527f, h5 = 0x9b05688c, h6 = 0x1f83d9ab, h7 = 0x5be0cd19;

    size_t orig_len = len;
    size_t padded = ((len + 9 + 63) / 64) * 64;
    std::vector<uint8_t> buf(padded, 0);
    std::memcpy(buf.data(), data, len);
    buf[len] = 0x80;
    uint64_t bit_len = static_cast<uint64_t>(orig_len) * 8;
    for (int i = 0; i < 8; ++i)
        buf[padded - 1 - i] = static_cast<uint8_t>(bit_len >> (i * 8));

    for (size_t offset = 0; offset < padded; offset += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t(buf[offset + i * 4]) << 24) |
                    (uint32_t(buf[offset + i * 4 + 1]) << 16) |
                    (uint32_t(buf[offset + i * 4 + 2]) << 8) |
                    uint32_t(buf[offset + i * 4 + 3]);
        for (int i = 16; i < 64; ++i)
            w[i] = gam1(w[i - 2]) + w[i - 7] + gam0(w[i - 15]) + w[i - 16];

        uint32_t a = h0, b = h1, c = h2, d = h3;
        uint32_t e = h4, f = h5, g = h6, h = h7;

        for (int i = 0; i < 64; ++i) {
            uint32_t t1 = h + sig1(e) + ch(e, f, g) + K[i] + w[i];
            uint32_t t2 = sig0(a) + maj(a, b, c);
            h = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        h0 += a; h1 += b; h2 += c; h3 += d;
        h4 += e; h5 += f; h6 += g; h7 += h;
    }

    std::array<uint8_t, 32> digest;
    auto store = [&](int idx, uint32_t val) {
        digest[idx * 4 + 0] = (val >> 24) & 0xFF;
        digest[idx * 4 + 1] = (val >> 16) & 0xFF;
        digest[idx * 4 + 2] = (val >> 8) & 0xFF;
        digest[idx * 4 + 3] = val & 0xFF;
    };
    store(0, h0); store(1, h1); store(2, h2); store(3, h3);
    store(4, h4); store(5, h5); store(6, h6); store(7, h7);
    return digest;
}

}  // namespace

std::string sha256_hex(const std::string& input) {
    auto digest = sha256_bytes(
        reinterpret_cast<const uint8_t*>(input.data()), input.size());
    std::ostringstream oss;
    for (auto b : digest)
        oss << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(b);
    return oss.str();
}

std::string hash_event(const Event& event) {
    // Canonical string matching Python's hashing.hash_event()
    std::ostringstream oss;
    oss << "{\"event_type\": \"" << event.event_type << "\""
        << ", \"payload_hash\": \"" << event.payload_hash << "\""
        << ", \"phase\": \"" << event.phase << "\""
        << ", \"request_id\": \"" << event.request_id << "\""
        << ", \"sequence_no\": " << event.sequence_no
        << ", \"service_name\": \"" << event.service_name << "\""
        << "}";
    return sha256_hex(oss.str());
}

std::string hash_trace(const std::vector<Event>& events) {
    // Cumulative hash matching Python's hashing.hash_trace()
    std::string running;
    for (const auto& evt : events) {
        std::string evt_hash = hash_event(evt);
        running += evt_hash;
        running = sha256_hex(running);
    }
    return running.empty() ? sha256_hex("") : running;
}

}  // namespace replay
