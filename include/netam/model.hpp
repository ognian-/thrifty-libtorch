#pragma once

#include <netam/kmer_sequence_encoder.hpp>
#include <netam/indep_rscnn_model.hpp>

#include <filesystem>
#include <fstream>

namespace netam {

class model {
 public:
  model(const std::filesystem::path& weights_path,
        const std::filesystem::path& yaml_path)
      : yaml_{YAML::LoadFile(yaml_path)},
        encoder_{yaml_["encoder_parameters"]},
        model_{{encoder_.kmer_count(), yaml_["model_hyperparameters"]}} {
    std::ifstream file(weights_path, std::ios::binary);
    std::vector<char> data{std::istreambuf_iterator<char>{file},
                           std::istreambuf_iterator<char>{}};
    auto loaded = torch::pickle_load(data);
    auto state_dict = loaded.toGenericDict();
    torch::NoGradGuard no_grad;

    for (auto& param : model_.named_parameters()) {
      std::string name = param.key();
      Assert(state_dict.contains(name));
      torch::Tensor loaded_tensor = state_dict.at(name).toTensor();
      param.value().copy_(loaded_tensor);
    }

    for (auto& buffer : model_.named_buffers()) {
      std::string name = buffer.key();
      Assert(state_dict.contains(name));
      torch::Tensor loaded_tensor = state_dict.at(name).toTensor();
      buffer.value().copy_(loaded_tensor);
    }
  }

  KmerSequenceEncoder& encoder() noexcept { return encoder_; }

  IndepRSCNNModel* operator->() noexcept { return &model_; }

 private:
  YAML::Node yaml_;
  KmerSequenceEncoder encoder_;
  IndepRSCNNModel model_;
};

}  // namespace netam
