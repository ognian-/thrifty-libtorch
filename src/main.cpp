#include <netam/model.hpp>
#include <netam/pcp_dataframe.hpp>

int main() {
  netam::model model{"../data/ThriftyHumV0.2-45-libtorch.pth",
                     "../data/ThriftyHumV0.2-45.yml"};

  torch::NoGradGuard no_grad;
  model->eval();

  netam::pcp_dataframe pcp_df{
      "../data/wyatt-10x-1p5m_pcp_2023-11-30_NI.first100.csv.gz"};

  std::string sequence_parent_heavy;

  std::size_t row_idx = 0;
  for (auto&& row : pcp_df.read()) {
    if (row_idx++ == 0) {
      continue;
    }
    std::size_t col_idx = 0;
    for (auto&& col : row) {
      if (col_idx++ == 3) {
        sequence_parent_heavy = col;
        break;
      }
    }
    break;
  }

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

  fmt::println("Done.");
  return 0;
}
