#pragma once

#include <netam/common.hpp>

#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <functional>

namespace netam {
class KmerSequenceEncoder {
 public:
  KmerSequenceEncoder(const YAML::Node& yaml)
      : kmer_length_{yaml["kmer_length"].as<std::size_t>()},
        site_count_{yaml["site_count"].as<std::size_t>()} {
    Assert(kmer_length_ % 2 == 1);
    overhang_length_ = (kmer_length_ - 1) / 2;

    // Generate all kmers
    all_kmers_ = generateKmers(kmer_length_);

    // Build kmer to index map
    for (std::size_t i = 0; i < all_kmers_.size(); ++i) {
      kmer_to_index_[all_kmers_[i]] = i;
    }
  }

  std::pair<torch::Tensor, torch::Tensor> encodeSequence(
      const std::string& sequence) {
    std::string upper_seq = toUpper(sequence);

    // Pad sequence with 'N's
    std::string padded_sequence = std::string(overhang_length_, 'N') +
                                  upper_seq +
                                  std::string(overhang_length_, 'N');

    // Encode kmers
    std::vector<std::int32_t> kmer_indices;
    for (std::size_t i = 0; i < site_count_; ++i) {
      if (i + kmer_length_ <= padded_sequence.length()) {
        std::string kmer = padded_sequence.substr(i, kmer_length_);
        auto it = kmer_to_index_.find(kmer);
        kmer_indices.push_back(
            it != kmer_to_index_.end()
                ? signed_cast(narrowing_cast<std::uint32_t>(it->second))
                : 0);
      } else {
        kmer_indices.push_back(0);
      }
    }

    torch::Tensor encoded = torch::tensor(kmer_indices, torch::kInt32);
    torch::Tensor wt_base_modifier = computeWtBaseModifier(upper_seq);

    return {encoded, wt_base_modifier};
  }

  std::size_t getKmerCount() const noexcept { return all_kmers_.size(); }

  std::size_t getKmerLength() const noexcept { return kmer_length_; }

  std::size_t getSiteCount() const noexcept { return site_count_; }

 private:
  const std::size_t kmer_length_;
  const std::size_t site_count_;
  std::size_t overhang_length_;
  std::vector<std::string> all_kmers_;
  std::unordered_map<std::string, std::size_t> kmer_to_index_;

  static constexpr const char* BASES = "ACGT";
  static constexpr float BIG = 30.0f;

  std::vector<std::string> generateKmers(std::size_t length) {
    std::vector<std::string> kmers;
    kmers.push_back("N");  // Placeholder for kmers with N

    // Generate all possible kmers of given length
    std::function<void(std::string, std::size_t)> generate =
        [&](std::string current, std::size_t pos) {
          if (pos == length) {
            kmers.push_back(current);
            return;
          }
          for (std::size_t i = 0; i < 4; ++i) {
            generate(current + BASES[i], pos + 1);
          }
        };

    generate("", 0);
    return kmers;
  }

  torch::Tensor computeWtBaseModifier(const std::string& parent) {
    torch::Tensor wt_base_modifier =
        torch::zeros({signed_cast(site_count_), 4});

    for (std::size_t i = 0; i < std::min(parent.length(), site_count_); ++i) {
      char base = parent[i];
      wt_base_modifier[signed_cast(i)][signed_cast(getBaseIndex(base))] = -BIG;
    }

    return wt_base_modifier;
  }

  std::size_t getBaseIndex(char base) {
    switch (base) {
      case 'A':
        return 0;
      case 'C':
        return 1;
      case 'G':
        return 2;
      case 'T':
        return 3;
      default:
        fail("Unknown base");
    }
  }

  std::string toUpper(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::toupper);
    return result;
  }
};
}  // namespace netam
