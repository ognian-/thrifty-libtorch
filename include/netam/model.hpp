#pragma once

#include <netam/kmer_sequence_encoder.hpp>
#include <netam/indep_rscnn_model.hpp>

#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include <filesystem>

namespace netam {

class model {
 public:
  model(const std::filesystem::path& weights_path,
        const std::filesystem::path& yaml_path) {
    YAML::Node yaml = YAML::LoadFile(yaml_path);
    std::println("encoder_class: {}", yaml["encoder_class"].as<std::string>());
  }

 private:
};

}  // namespace netam
