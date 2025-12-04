#pragma once

#include <netam/common.hpp>

namespace netam {

inline void TestAssert(bool condition, std::source_location location =
                                           std::source_location::current()) {
  if (!condition) {
    fail("Assertion failed at {}:{}:{} in function '{}'", location.file_name(),
         location.line(), location.column(), location.function_name());
  }
}

}  // namespace netam

using namespace netam;
