# My Master's Degree Thesis for Chinese Grammatical Error Correction
Chinese proofreading based on single model inference, ensemble decoding, re-ranking mechanism and multi-channel fusion framework.

## Setting Up
TODO

## Codes Related to Chapter 4th

### Pre-training Word Embeddings

#### Pre-training Chinese Word2vec Embeddings
TODO

#### Pre-training Chinese Wang2vec Embeddings
TODO

#### Pre-training Chinese Cw2vec Embeddings
TODO

### Training a Single Model
Use `one_script_to_run_all.sh` to train the model by specifying model architecture, model level, which pre-trained embeddings to use and data fusion mode.

### Data Cleaning Experiments
Use `tune_long_low_high.sh` to determine filtering thresholds including `long`, `low` and `high` parameters.


## For Chapter 5th

### Training Single Models with Different Random Seeds
Use `tune_random_seed.sh` to train single models with different random number seeds.

### Pre-training 5-Gram Language Models
TODO

### Ensemble Decoding + Re-ranking Mechanism
Reproduce experiments of chapter 5 of my master's dissertation with `rerank_experiment.sh`. The `rerank_experiment.sh` shell script trains the re-ranker calling `train_reranker.sh` firstly, and then applies the re-ranking mechanism via `run_trained_model.sh`. Run `rerank_experiment.sh` script directly in the terminal for more details.


## For Chapter 6

### Multi-channel Fusion Framework + Re-ranking Mechanism
Multi-channel fusion and re-ranking using `multi_channel_fusion_experiment.sh`. The `multi_channel_fusion_experiment.sh` bash script trains the re-ranker components calling `train_multi_channel_fusion_reranker.sh` firstly, and then multi-channel fusion and applies the re-ranking mechanism via `multi_channel_fusion.sh`. Run `multi_channel_fusion_experiment.sh` script directly in the terminal for more details.

### Reproduce Experiments of Chapter 6 of My Master's Dissertation
Reproduce experiments of chapter 6 of my master's dissertation with `all_experiments_multi_channel_fusion.sh`.

