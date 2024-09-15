#!/usr/bin/env bash

find src -name "*.cpp" -o -name "*.hpp" | xargs clang-format -style=file -i
