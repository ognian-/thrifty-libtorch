#pragma once
#include <torch/torch.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <functional>

namespace netam {
class KmerSequenceEncoder {
 public:
  KmerSequenceEncoder(int kmer_length, int site_count)
      : kmer_length_(kmer_length), site_count_(site_count) {
    assert(kmer_length_ % 2 == 1);
    overhang_length_ = (kmer_length_ - 1) / 2;

    // Generate all kmers
    all_kmers_ = generateKmers(kmer_length_);

    // Build kmer to index map
    for (size_t i = 0; i < all_kmers_.size(); ++i) {
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
    std::vector<int32_t> kmer_indices;
    for (int i = 0; i < site_count_; ++i) {
      if (i + kmer_length_ <= padded_sequence.length()) {
        std::string kmer = padded_sequence.substr(i, kmer_length_);
        auto it = kmer_to_index_.find(kmer);
        kmer_indices.push_back(it != kmer_to_index_.end() ? it->second : 0);
      } else {
        kmer_indices.push_back(0);
      }
    }

    torch::Tensor encoded = torch::tensor(kmer_indices, torch::kInt32);
    torch::Tensor wt_base_modifier = computeWtBaseModifier(upper_seq);

    return {encoded, wt_base_modifier};
  }

  // Getter methods
  int getKmerCount() const { return all_kmers_.size(); }

  int getKmerLength() const { return kmer_length_; }

  int getSiteCount() const { return site_count_; }

 private:
  int kmer_length_;
  int site_count_;
  int overhang_length_;
  std::vector<std::string> all_kmers_;
  std::unordered_map<std::string, int> kmer_to_index_;

  static constexpr const char* BASES = "ACGT";
  static constexpr float BIG = 30.0f;

  std::vector<std::string> generateKmers(int length) {
    std::vector<std::string> kmers;
    kmers.push_back("N");  // Placeholder for kmers with N

    // Generate all possible kmers of given length
    std::function<void(std::string, int)> generate = [&](std::string current,
                                                         int pos) {
      if (pos == length) {
        kmers.push_back(current);
        return;
      }
      for (int i = 0; i < 4; ++i) {
        generate(current + BASES[i], pos + 1);
      }
    };

    generate("", 0);
    return kmers;
  }

  torch::Tensor computeWtBaseModifier(const std::string& parent) {
    torch::Tensor wt_base_modifier = torch::zeros({site_count_, 4});

    for (int i = 0;
         i < std::min(static_cast<int>(parent.length()), site_count_); ++i) {
      char base = parent[i];
      int base_idx = getBaseIndex(base);
      if (base_idx >= 0) {
        wt_base_modifier[i][base_idx] = -BIG;
      }
    }

    return wt_base_modifier;
  }

  int getBaseIndex(char base) {
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
        return -1;
    }
  }

  std::string toUpper(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::toupper);
    return result;
  }
};
}  // namespace netam
