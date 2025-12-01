#pragma once

#include <fmt/format.h>
#include <iostream>
#include <source_location>
#include <limits>

namespace netam {

[[noreturn]] inline void fail(std::string_view fmt, auto&&... args) {
  fmt::println("Failed: {}", fmt::vformat(fmt, fmt::make_format_args(args...)));
  std::cout << std::flush;
  throw std::runtime_error{std::vformat(fmt, std::make_format_args(args...))};
}

inline void Assert(bool condition, std::source_location location =
                                       std::source_location::current()) {
  if (!condition) {
    fail("Assertion failed at {}:{}:{} in function '{}'", location.file_name(),
         location.line(), location.column(), location.function_name());
  }
}

template <typename To, typename From>
  requires(std::integral<From> and std::integral<To>)
constexpr To checked_cast(From x) {
  if constexpr (std::is_signed_v<From> == std::is_signed_v<To>) {
    if (x < std::numeric_limits<To>::min()) {
      goto err;
    }
    if (x > std::numeric_limits<To>::max()) {
      goto err;
    }
  } else if constexpr (std::is_unsigned_v<To>) {
    if (x < 0) {
      goto err;
    }
    // TODO
  } else {
    // TODO
  }
  return static_cast<To>(x);
err:
  fail("cast out of bounds");
}

template <typename To, typename From>
  requires((sizeof(To) < sizeof(From)) and
           (std::is_signed_v<From> == std::is_signed_v<To>))
constexpr To narrowing_cast(From x) {
  return checked_cast<To>(x);
}

template <typename From>
  requires(std::is_unsigned_v<From>)
constexpr std::make_signed_t<From> signed_cast(From x) {
  return checked_cast<std::make_signed_t<From>>(x);
}

}  // namespace netam
