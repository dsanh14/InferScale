cmake_minimum_required(VERSION 3.16)
project(replay_validator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Main executable
add_executable(replay_validator
    src/main.cpp
    src/replay_validator.cpp
    src/event_parser.cpp
    src/state_hasher.cpp
)

target_include_directories(replay_validator PRIVATE include)

# Test executable
add_executable(replay_validator_test
    tests/replay_validator_test.cpp
    src/replay_validator.cpp
    src/event_parser.cpp
    src/state_hasher.cpp
)

target_include_directories(replay_validator_test PRIVATE include)
