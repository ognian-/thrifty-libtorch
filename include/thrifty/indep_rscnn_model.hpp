#pragma once
#include <torch/torch.h>

class IndepRSCNNModelImpl : public torch::nn::Module {
 public:
  IndepRSCNNModelImpl(int kmer_count, int kmer_length, int embedding_dim,
                      int filter_count, int kernel_size,
                      double dropout_prob = 0.1)
      : kmer_count_(kmer_count),
        kmer_length_(kmer_length),
        kernel_size_(kernel_size),

        // Calculate padding for "same" convolution
        // For same padding: padding = (kernel_size - 1) / 2
        padding_((kernel_size - 1) / 2),

        // R component layers
        r_kmer_embedding_(
            register_module("r_kmer_embedding",
                            torch::nn::Embedding(kmer_count, embedding_dim))),
        r_conv_(register_module(
            "r_conv",
            torch::nn::Conv1d(torch::nn::Conv1dOptions(
                                  embedding_dim, filter_count, kernel_size)
                                  .padding(padding_)))),
        r_dropout_(
            register_module("r_dropout", torch::nn::Dropout(dropout_prob))),
        r_linear_(
            register_module("r_linear", torch::nn::Linear(filter_count, 1))),

        // S component layers
        s_kmer_embedding_(
            register_module("s_kmer_embedding",
                            torch::nn::Embedding(kmer_count, embedding_dim))),
        s_conv_(register_module(
            "s_conv",
            torch::nn::Conv1d(torch::nn::Conv1dOptions(
                                  embedding_dim, filter_count, kernel_size)
                                  .padding(padding_)))),
        s_dropout_(
            register_module("s_dropout", torch::nn::Dropout(dropout_prob))),
        s_linear_(
            register_module("s_linear", torch::nn::Linear(filter_count, 4))) {}

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
  int kmer_count_;
  int kmer_length_;
  int kernel_size_;
  int padding_;

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

TORCH_MODULE(IndepRSCNNModel);
