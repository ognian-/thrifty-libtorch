#include <netam/model.hpp>

#include <print>
#include <iostream>
#include <fstream>

/*
encoder_class: KmerSequenceEncoder
encoder_parameters:
  kmer_length: 3
  site_count: 500
model_class: IndepRSCNNModel
model_hyperparameters:
  dropout_prob: 0.2
  embedding_dim: 7
  filter_count: 16
  kernel_size: 9
  kmer_length: 3
serialization_version: 0
training_hyperparameters:
  learning_rate: 0.001
  min_learning_rate: 1.0e-06
  weight_decay: 1.0e-06
*/

std::vector<char> read_file(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  return std::vector<char>(std::istreambuf_iterator<char>(file),
                           std::istreambuf_iterator<char>());
}

int main() {
  netam::model m{"ThriftyHumV0.2-45-libtorch.pth", "ThriftyHumV0.2-45.yml"};

  // Initialize encoder
  int kmer_length = 3;
  int site_count = 500;
  netam::KmerSequenceEncoder encoder(kmer_length, site_count);

  // Initialize model
  int kmer_count = encoder.getKmerCount();
  int embedding_dim = 7;
  int filter_count = 16;
  int kernel_size = 9;
  double dropout_prob = 0.2;

  netam::IndepRSCNNModel model(kmer_count, kmer_length, embedding_dim,
                               filter_count, kernel_size, dropout_prob);

  std::println("------");
  std::cout << model << std::endl;  // Shows registered modules
  std::println("------");

  try {
    auto data = read_file("ThriftyHumV0.2-45-libtorch.pth");
    auto loaded = torch::pickle_load(data);
    auto state_dict = loaded.toGenericDict();

    // Print available keys
    std::cout << "State dict keys:\n";
    for (const auto& item : state_dict) {
      std::string key = item.key().toStringRef();
      torch::Tensor tensor = item.value().toTensor();
      std::cout << "  " << key << ": " << tensor.sizes() << std::endl;
    }

    // Load parameters manually
    torch::NoGradGuard no_grad;

    for (auto& param : model.named_parameters()) {
      std::string name = param.key();

      if (state_dict.contains(name)) {
        torch::Tensor loaded_tensor = state_dict.at(name).toTensor();
        param.value().copy_(loaded_tensor);
        std::cout << "Loaded: " << name << std::endl;
      } else {
        std::cerr << "Warning: " << name << " not found in state_dict"
                  << std::endl;
      }
    }

    // Also load buffers (e.g., batch norm running mean/var)
    for (auto& buffer : model.named_buffers()) {
      std::string name = buffer.key();

      if (state_dict.contains(name)) {
        torch::Tensor loaded_tensor = state_dict.at(name).toTensor();
        buffer.value().copy_(loaded_tensor);
        std::cout << "Loaded buffer: " << name << std::endl;
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }

  // Set to evaluation mode
  model.eval();

  // Process a sequence
  std::string sequence_parent_heavy =
      "CAGGTGCAGCTGGTGGAGTCTGGGGGAGGCGTGGTCCAGCCTGGGAGGTCCCTGAGACTCTCCTGTGCAGCG"
      "TCTGGATTCACCTTCAGTAGCTCTGGCATGCACTGGGTCCGCCAGGCTCCAGGCAAGGGGCTGGAGTGGGTG"
      "GCAGTTATATGGTATGATGGAAGTAATAAATATTATGCAGACTCCGTGAAGGGCCGATTCACCATCTCCAGA"
      "GACAATTCCAAGAACACGGTGTATCTTCAAATGAACAGCCTAAGAGCCGAGGACACGGCTGTGTATTACTGT"
      "GCGAGAGAGGGGCACAGTAACTACCCCTACTACTACTACTACATGGACGTCTGGGGCAAAGGGACCACGGTC"
      "ACCGTCTCCTCA";
  auto [encoded, wt_modifier] = encoder.encodeSequence(sequence_parent_heavy);

  // Create mask (all 1s for valid positions)
  torch::Tensor mask = torch::ones({1, site_count}, torch::kBool);

  // Add batch dimension
  encoded = encoded.unsqueeze(0);
  wt_modifier = wt_modifier.unsqueeze(0);

  // Forward pass
  torch::NoGradGuard no_grad;
  auto [rates, csp_logits] = model.forward(encoded, mask, wt_modifier);

  std::cout << "Rates shape: " << rates.sizes() << std::endl;
  std::cout << "CSP logits shape: " << csp_logits.sizes() << std::endl;

  std::println("Done.");
  return 0;
}
