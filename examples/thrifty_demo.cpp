// C++ implementation of thrifty_demo.ipynb
// Demonstrates rate prediction and conditional substitution probabilities
// (CSPs) from the ThriftyHumV0.2-45 model

#include <netam/common.hpp>
#include <netam/model.hpp>
#include <netam/pcp_dataframe.hpp>

#include <netam/include-matplot.hpp>

#include <vector>

int main() {
  // Load the model (equivalent to pretrained.load("ThriftyHumV0.2-45"))
  netam::model model{"../data/ThriftyHumV0.2-45-libtorch.pth",
                     "../data/ThriftyHumV0.2-45.yml"};

  torch::NoGradGuard no_grad;
  model->eval();

  // Load the PCP dataframe
  netam::pcp_dataframe pcp_df{
      "../data/wyatt-10x-1p5m_pcp_2023-11-30_NI.first100.csv.gz"};

  // Extract the first 3 sequences (parent_heavy column)
  // This mirrors: seqs = pcp_df.loc[:2, "parent_heavy"]
  std::vector<std::string> sequences;
  constexpr std::size_t num_sequences = 3;
  constexpr std::size_t parent_heavy_col = 3;

  std::size_t row_idx = 0;
  for (auto&& row : pcp_df.read()) {
    if (row_idx == 0) {
      // Skip header
      row_idx++;
      continue;
    }

    std::size_t col_idx = 0;
    for (auto&& col : row) {
      if (col_idx == parent_heavy_col) {
        sequences.emplace_back(col);
        break;
      }
      col_idx++;
    }

    if (sequences.size() >= num_sequences) {
      break;
    }
    row_idx++;
  }

  fmt::println("Loaded {} sequences", sequences.size());
  fmt::println("Sequence 0: {}", sequences[0]);

  // Run inference for each sequence and collect rates and CSP logits
  std::vector<torch::Tensor> rates_list;
  std::vector<torch::Tensor> csp_logits_list;

  for (const auto& seq : sequences) {
    auto [encoded, wt_modifier] = model.encoder().encode_sequence(seq);

    // Create mask (all 1s for valid positions)
    torch::Tensor mask = torch::ones(
        {1, netam::signed_cast(model.encoder().site_count())}, torch::kBool);

    // Add batch dimension
    encoded = encoded.unsqueeze(0);
    wt_modifier = wt_modifier.unsqueeze(0);

    // Forward pass
    auto [rates, csp_logits] = model->forward(encoded, mask, wt_modifier);

    // Remove batch dimension for storage
    rates_list.push_back(rates.squeeze(0));
    csp_logits_list.push_back(csp_logits.squeeze(0));
  }

  // Plot rates for the three sequences (first 350 sites)
  // Mirrors the notebook's rate plot
  constexpr std::size_t plot_sites = 350;

  using namespace matplot;

  auto fig1 = figure(true);
  fig1->size(1500, 400);
  hold(on);

  for (std::size_t i = 0; i < sequences.size(); i++) {
    std::vector<double> rates_vec;
    auto rates_accessor = rates_list[i].accessor<float, 1>();

    std::size_t sites_to_plot =
        std::min(plot_sites, static_cast<std::size_t>(rates_accessor.size(0)));
    for (std::size_t j = 0; j < sites_to_plot; j++) {
      rates_vec.push_back(
          static_cast<double>(rates_accessor[netam::signed_cast(j)]));
    }

    auto p = plot(rates_vec);
    p->display_name("Sequence " + std::to_string(i));
  }

  xlabel("Site");
  ylabel("Rate");
  legend();
  title("Rate estimates for three sequences");
  save("rates_plot.png");
  fmt::println("Saved rates plot to rates_plot.png");

  // Compute CSPs using softmax (for the first sequence)
  // Mirrors: csps = torch.softmax(csp_logitss[0], dim=1)
  auto csps = torch::softmax(csp_logits_list[0], /*dim=*/1);

  // Create heatmap of CSPs for first 50 sites
  // The notebook shows CSPs as a 4xN matrix (A, C, G, T rows)
  constexpr std::size_t heatmap_sites = 50;
  auto csps_accessor = csps.accessor<float, 2>();

  std::size_t actual_sites =
      std::min(heatmap_sites, static_cast<std::size_t>(csps_accessor.size(0)));

  // Prepare data for heatmap: 4 rows (A, C, G, T) x N columns (sites)
  std::vector<std::vector<double>> heatmap_data(4);
  for (auto& row : heatmap_data) {
    row.resize(actual_sites);
  }

  for (std::size_t site = 0; site < actual_sites; site++) {
    for (std::size_t base = 0; base < 4; base++) {
      heatmap_data[base][site] = static_cast<double>(
          csps_accessor[netam::signed_cast(site)][netam::signed_cast(base)]);
    }
  }

  auto fig2 = figure(true);
  fig2->size(2000, 300);

  auto h = heatmap(heatmap_data);

  // Set y-axis labels to nucleotides
  std::vector<std::string> y_labels = {"A", "C", "G", "T"};
  yticks({1, 2, 3, 4});
  yticklabels(y_labels);

  xlabel("Sites");
  ylabel("Nucleotides");
  title("Conditional Substitution Probabilities (first 50 sites)");
  colorbar();
  save("csp_heatmap.png");
  fmt::println("Saved CSP heatmap to csp_heatmap.png");

  fmt::println("Done.");
  return 0;
}
