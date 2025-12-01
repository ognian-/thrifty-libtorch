#pragma once

#include <netam/generator.hpp>

#include <zlib.h>

#include <filesystem>
#include <ranges>
#include <string_view>
#include <array>
#include <memory>

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
    auto close = [](::gzFile x) static {
      if (x != nullptr) {
        ::gzclose(x);
      }
    };
    std::unique_ptr<std::remove_pointer_t<gzFile>, decltype(close)> file{
        ::gzopen(path.c_str(), "rb"), close};

    if (file == nullptr) {
      fail("Can't open gzip file");
    }

    std::string buffer;
    std::array<char, 4096> chunk;

    for (int bytes = ::gzread(file.get(), chunk.data(), chunk.size());
         bytes > 0; bytes = ::gzread(file.get(), chunk.data(), chunk.size())) {
      buffer.append(chunk.data(), unsigned_cast(bytes));

      std::size_t pos;
      while ((pos = buffer.find('\n')) != std::string::npos) {
        co_yield buffer.substr(0, pos);
        buffer.erase(0, pos + 1);
      }
    }

    if (not buffer.empty()) {
      co_yield buffer;
    }
  }

  std::filesystem::path path_;
};

}  // namespace netam
