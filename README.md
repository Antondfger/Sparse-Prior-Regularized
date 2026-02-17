# Mitigating Popularity Bias in Sequential Recommendations with Sparse Prior-Regularized Objective
## Abstract
Popularity bias is a well-known challenge in recommender systems, often leading to the over-exposure of popular items and reduced catalog coverage. In this work, we propose sparse prior-regularized objectives for mitigating popularity bias in sequential recommendations. Our approach introduces sparsity into the model outputs during training using entmax, resulting in adaptive, context-dependent competition among items. In addition, we incorporate an explicit regularization with respect to an item popularity prior, which admits a variational interpretation and leads to a principled form of logit adjustment. Together, these components yield a unified training objective that balances sparse competition with prior-aware regularization. Experiments on multiple benchmark datasets show that the proposed approach consistently reduces popularity bias, while preserving or slightly improving relevance of recommendations.

## Usage
Install requirements:
```sh
pip install -r requirements.txt
```
Specify environment variables:
```sh
# path to the project
export PATH4SEQ="/your/path"
# path to the raw data
export RECSYS_DATA_PATH="/your/path"
``
```

```sh
# PR-Entmax on Tmall dataset
python dl.py datasets_info=Tmall dataset_params.num_negatives=50000 model_params.num_blocks=2 model_params.num_heads=4 model_params.dropout_rate=0.1 model_params.hidden_units=256 type_loss=entmax temp=1.71 adj=0.05 seqrec_module.type_adj=logit

# Entmax on Tmall dataset
python dl.py datasets_info=Tmall dataset_params.num_negatives=50000 model_params.num_blocks=2 model_params.num_heads=4 model_params.dropout_rate=0.1 model_params.hidden_units=256 type_loss=entmax temp=1.71

# Logit Adjustment on Tmall dataset
python dl.py datasets_info=Tmall dataset_params.num_negatives=50000 model_params.num_blocks=3 model_params.num_heads=4 model_params.dropout_rate=0.1 model_params.hidden_units=256 type_loss=ce temp=1 seqrec_module.type_adj=logit adj=0.4

# Margin Adjustment on Tmall dataset
python dl.py datasets_info=Tmall dataset_params.num_negatives=50000 model_params.num_blocks=3 model_params.num_heads=4 model_params.dropout_rate=0.1 model_params.hidden_units=256 type_loss=ce temp=1 seqrec_module.type_adj=margin adj=0.01

# Cross-Entropy on Tmall dataset
python dl.py datasets_info=Tmall dataset_params.num_negatives=50000 model_params.num_blocks=3 model_params.num_heads=4 model_params.dropout_rate=0.1 model_params.hidden_units=256 type_loss=ce temp=1

#Parameter tuning for Tmall with ce loss
python optuna_rec.py datasets_info=Tmall type_loss=ce --multirun

#Parameter tuning for Tmall with entmax loss
python optuna_rec.py datasets_info=Tmall type_loss=entmax --multirun

```
