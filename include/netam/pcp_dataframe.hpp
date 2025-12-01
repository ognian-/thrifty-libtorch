#pragma once

#include <netam/generator.hpp>

#include <filesystem>
#include <ranges>
#include <string_view>
#include <fstream>

namespace netam {

class pcp_dataframe {
 public:
  pcp_dataframe(const std::filesystem::path& csv_gz_path)
      : path_{csv_gz_path} {}

  auto read() {
    return std::ranges::owning_view(lines(path_)) |
           std::views::transform([](std::string_view line) {
             return line | std::views::split(',') |
                    std::views::transform(
                        [](auto&& x) { return std::string_view(x); });
           });
  }

 private:
  static generator<std::string> lines(const std::filesystem::path& path) {
    std::ifstream is{path};
    std::string line;
    while (std::getline(is, line)) {
      co_yield line;
    }
  }

  std::filesystem::path path_;
};

}  // namespace netam
