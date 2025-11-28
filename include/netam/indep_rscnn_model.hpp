#pragma once

#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

namespace netam {

class IndepRSCNNParams {
 public:
  IndepRSCNNParams(std::size_t kmer_count, const YAML::Node& yaml)
      : kmer_count_{kmer_count},
        kmer_length_{yaml["kmer_length"].as<std::size_t>()},
        embedding_dim_{yaml["embedding_dim"].as<std::size_t>()},
        filter_count_{yaml["filter_count"].as<std::size_t>()},
        kernel_size_{yaml["kernel_size"].as<std::size_t>()},
        dropout_prob_{yaml["dropout_prob"].as<double>()} {}

  IndepRSCNNParams(const IndepRSCNNParams&) = default;

  std::size_t kmer_count() const noexcept { return kmer_count_; }

  std::size_t kmer_length() const noexcept { return kmer_length_; }

  std::size_t embedding_dim() const noexcept { return embedding_dim_; }

  std::size_t filter_count() const noexcept { return filter_count_; }

  std::size_t kernel_size() const noexcept { return kernel_size_; }

  double dropout_prob() const noexcept { return dropout_prob_; }

 private:
  const std::size_t kmer_count_;
  const std::size_t kmer_length_;
  const std::size_t embedding_dim_;
  const std::size_t filter_count_;
  const std::size_t kernel_size_;
  const double dropout_prob_;
};

class IndepRSCNNModel : public torch::nn::Module {
 public:
  IndepRSCNNModel(const IndepRSCNNParams& params)
      :

        // Calculate padding for "same" convolution
        // For same padding: padding = (kernel_size - 1) / 2
        padding_{(params.kernel_size() - 1) / 2},

        // R component layers
        r_kmer_embedding_(register_module(
            "r_kmer_embedding",
            torch::nn::Embedding(params.kmer_count(), params.embedding_dim()))),
        r_conv_(register_module(
            "r_conv", torch::nn::Conv1d(torch::nn::Conv1dOptions(
                                            signed_cast(params.embedding_dim()),
                                            signed_cast(params.filter_count()),
                                            signed_cast(params.kernel_size()))
                                            .padding(padding_)))),
        r_dropout_(register_module("r_dropout",
                                   torch::nn::Dropout(params.dropout_prob()))),
        r_linear_(register_module("r_linear",
                                  torch::nn::Linear(params.filter_count(), 1))),

        // S component layers
        s_kmer_embedding_(register_module(
            "s_kmer_embedding",
            torch::nn::Embedding(params.kmer_count(), params.embedding_dim()))),
        s_conv_(register_module(
            "s_conv", torch::nn::Conv1d(torch::nn::Conv1dOptions(
                                            signed_cast(params.embedding_dim()),
                                            signed_cast(params.filter_count()),
                                            signed_cast(params.kernel_size()))
                                            .padding(padding_)))),
        s_dropout_(register_module("s_dropout",
                                   torch::nn::Dropout(params.dropout_prob()))),
        s_linear_(register_module(
            "s_linear", torch::nn::Linear(params.filter_count(), 4))) {}

  std::pair<torch::Tensor, torch::Tensor> forward(
      torch::Tensor encoded_parents, torch::Tensor masks,
      torch::Tensor wt_base_modifier) {
    // Process R component
    auto r_kmer_embeds = r_kmer_embedding_->forward(encoded_parents);
    r_kmer_embeds = r_kmer_embeds.permute({0, 2, 1});  // [B, E, L]
    auto r_conv_out = torch::relu(r_conv_->forward(r_kmer_embeds));
    r_conv_out = r_dropout_->forward(r_conv_out);
    r_conv_out = r_conv_out.permute({0, 2, 1});  // [B, L, F]

    auto log_rates = r_linear_->forward(r_conv_out).squeeze(-1);  // [B, L]
    auto rates = torch::exp(log_rates * masks);

    // Process S component
    auto s_kmer_embeds = s_kmer_embedding_->forward(encoded_parents);
    s_kmer_embeds = s_kmer_embeds.permute({0, 2, 1});  // [B, E, L]
    auto s_conv_out = torch::relu(s_conv_->forward(s_kmer_embeds));
    s_conv_out = s_dropout_->forward(s_conv_out);
    s_conv_out = s_conv_out.permute({0, 2, 1});  // [B, L, F]

    auto csp_logits = s_linear_->forward(s_conv_out);  // [B, L, 4]
    csp_logits = csp_logits * masks.unsqueeze(-1);
    csp_logits = csp_logits + wt_base_modifier;

    return {rates, csp_logits};
  }

  void adjustRateBiasBy(double log_adjustment_factor) {
    torch::NoGradGuard no_grad;
    r_linear_->bias.data() += log_adjustment_factor;
  }

 private:
  const std::size_t padding_;

  // R component layers
  torch::nn::Embedding r_kmer_embedding_;
  torch::nn::Conv1d r_conv_;
  torch::nn::Dropout r_dropout_;
  torch::nn::Linear r_linear_;

  // S component layers
  torch::nn::Embedding s_kmer_embedding_;
  torch::nn::Conv1d s_conv_;
  torch::nn::Dropout s_dropout_;
  torch::nn::Linear s_linear_;
};

}  // namespace netam
