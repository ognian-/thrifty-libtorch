#include <netam/model.hpp>

int main() {
  netam::model model{"ThriftyHumV0.2-45-libtorch.pth", "ThriftyHumV0.2-45.yml"};

  torch::NoGradGuard no_grad;
  model->eval();

  // Process a sequence
  std::string sequence_parent_heavy =
      "CAGGTGCAGCTGGTGGAGTCTGGGGGAGGCGTGGTCCAGCCTGGGAGGTCCCTGAGACTCTCCTGTGCAGCG"
      "TCTGGATTCACCTTCAGTAGCTCTGGCATGCACTGGGTCCGCCAGGCTCCAGGCAAGGGGCTGGAGTGGGTG"
      "GCAGTTATATGGTATGATGGAAGTAATAAATATTATGCAGACTCCGTGAAGGGCCGATTCACCATCTCCAGA"
      "GACAATTCCAAGAACACGGTGTATCTTCAAATGAACAGCCTAAGAGCCGAGGACACGGCTGTGTATTACTGT"
      "GCGAGAGAGGGGCACAGTAACTACCCCTACTACTACTACTACATGGACGTCTGGGGCAAAGGGACCACGGTC"
      "ACCGTCTCCTCA";
  auto [encoded, wt_modifier] =
      model.encoder().encode_sequence(sequence_parent_heavy);

  // Create mask (all 1s for valid positions)
  torch::Tensor mask = torch::ones(
      {1, netam::signed_cast(model.encoder().site_count())}, torch::kBool);

  // Add batch dimension
  encoded = encoded.unsqueeze(0);
  wt_modifier = wt_modifier.unsqueeze(0);

  // Forward pass
  auto [rates, csp_logits] = model->forward(encoded, mask, wt_modifier);

  std::cout << "Rates shape: " << rates.sizes() << std::endl;
  std::cout << "CSP logits shape: " << csp_logits.sizes() << std::endl;

  std::println("Done.");
  return 0;
}
