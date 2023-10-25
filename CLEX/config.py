{
  "architectures": [
    "LlamaForCausalLM"
  ],
  "auto_map": {
    "AutoConfig": "configuration_clex.CLEXLlamaConfig",
    "AutoModelForCausalLM": "modeling_llama.LlamaForCausalLM"
  },
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "pad_token_id": 0,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "tie_word_embeddings": false,
  "use_cache": true,
  "vocab_size": 32000,
  "log_scale": false,
  "use_flashattn": true,
  "rope_scaling": {
    "type": "clex",
    "max_factor": 16,
    "param_factor": 1,
  }
}