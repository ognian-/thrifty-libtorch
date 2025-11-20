#include <thrifty/model.hpp>

#include <print>
#include <iostream>

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

int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;

  // Initialize encoder
  int kmer_length = 3;
  int site_count = 500;
  KmerSequenceEncoder encoder(kmer_length, site_count);

  // Initialize model
  int kmer_count = encoder.getKmerCount();
  int embedding_dim = 7;
  int filter_count = 16;
  int kernel_size = 9;
  double dropout_prob = 0.2;

  IndepRSCNNModel model(kmer_count, kmer_length, embedding_dim, filter_count,
                        kernel_size, dropout_prob);

  // Load pre-trained weights
  torch::load(model, "ThriftyHumV0.2-45.pth");

  // Set to evaluation mode
  model->eval();

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
  auto [rates, csp_logits] = model->forward(encoded, mask, wt_modifier);

  std::cout << "Rates shape: " << rates.sizes() << std::endl;
  std::cout << "CSP logits shape: " << csp_logits.sizes() << std::endl;

  std::println("Done.");
  return 0;
}
