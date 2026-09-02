# Mitigating Popularity Bias in Sequential Recommendations with Sparse Prior-Regularized Objective


[Anton Pembek](https://scholar.google.com/citations?user=1oCtv4QAAAAJ) ·
[Anton Klenitskiy](https://scholar.google.com/citations?user=eGTslO8AAAAJ) ·
Alexander Savchenko ·
[Alexey Vasilev](https://scholar.google.com/citations?user=4vb0JIwAAAAJ)

---

> Popularity bias is a well-known challenge in recommender systems, often leading to the over-exposure of popular items and reduced catalog coverage. In this work, we propose PR-entmax, a sparse prior-regularized objective that makes popularity correction context-dependent. We derive our method from a generalized variational principle in which an item popularity prior is incorporated directly into a sparse probability mapping, yielding a popularity-aware active support: the set of items receiving non-zero probabilities and hence non-zero gradients is determined jointly by contextual relevance and item popularity. As a result, the prior affects not only item probabilities, but also which items participate in gradient updates during training. The method requires no architectural changes and can be integrated into existing training pipelines by modifying only the loss function. At inference time, the prior term is removed, and items are ranked by the learned contextual relevance alone. Experiments across multiple benchmark datasets and sequential recommendation architectures show that PR-entmax consistently reduces popularity bias while preserving or improving recommendation relevance, advancing the relevance-debiasing trade-off.

<p align="center">
  <img src="images/p_entmax_figure_2_2.png" width="900">
</p>

<p align="center">
  <em>
    Illustration of the PR-entmax training pipeline. For each item, the contextual logit produced by the backbone model is combined with the corresponding popularity prior. The resulting scores are transformed by entmax into a sparse, popularity-aware output distribution. The active support of this distribution determines which items receive non-zero gradients during training.
  </em>
</p>

---

## Method

PR-entmax introduces a **popularity-aware sparse training objective**.

For each item, the backbone produces a contextual relevance score, which is combined with a fixed popularity prior:

```math
s_i^{\mathrm{train}}(x) = z_i(x) + \gamma \log q_i
```

where:

- `z_i(x)` — contextual logit produced by the recommendation model;
- `q_i` — item popularity prior estimated from the training data;
- `γ` — strength of the popularity correction;
- `θ` — temperature parameter;
- `α` — entmax sparsity parameter.

The adjusted scores are transformed using entmax:

```math
p^*(z)
=
\mathrm{entmax}_{\alpha}
\left(
\frac{z + \gamma \log q}{\theta}
\right)
```

Unlike softmax, entmax assigns exactly zero probability to part of the catalog. As a result, the popularity prior affects not only item probabilities, but also **which items receive non-zero gradients during training**.

At inference time, the popularity prior is removed and items are ranked using only the learned contextual relevance:

```math
s_i^{\mathrm{inference}}(x) = z_i(x)
```
## Results

Experimental results on **SASRec** for all considered approaches. **Bold** numbers mark the best model, <u>underlined</u> numbers mark the second best. Asterisk (<sup>*</sup>) denotes a statistically significant improvement over *Cross-Entropy* for the best model.

| Dataset | Method | NDCG@10 | NDCG@100 | Novelty@10 | Novelty@100 | Coverage@10 | Coverage@100 |
|---|---|---:|---:|---:|---:|---:|---:|
| **ML1M** | *Cross-Entropy* | 0.047 | <u>0.102</u> | 0.274 | 0.308 | 0.463 | 0.835 |
|  | *IPW* | 0.047 | 0.100 | 0.290 | 0.321 | 0.496 | 0.871 |
|  | *PD* | 0.040 | 0.095 | <u>0.306</u> | <u>0.327</u> | <u>0.514</u> | 0.851 |
|  | *Pop Sampling* | <u>0.049</u> | <u>0.102</u> | 0.286 | 0.319 | 0.489 | <u>0.872</u> |
|  | *Logit Adjustment* | 0.047 | 0.100 | 0.291 | 0.321 | 0.496 | 0.869 |
|  | *PR-entmax* | **0.058**<sup>*</sup> | **0.103** | **0.331**<sup>*</sup> | **0.411**<sup>*</sup> | **0.600**<sup>*</sup> | **0.993**<sup>*</sup> |
| **Tmall** | *Cross-Entropy* | <u>0.274</u> | <u>0.305</u> | 0.657 | 0.664 | 0.351 | 0.649 |
|  | *IPW* | 0.265 | 0.296 | 0.704 | 0.704 | 0.431 | 0.722 |
|  | *PD* | 0.249 | 0.279 | 0.705 | 0.713 | 0.332 | 0.566 |
|  | *Pop Sampling* | 0.272 | 0.301 | **0.729**<sup>*</sup> | **0.755**<sup>*</sup> | <u>0.432</u> | <u>0.758</u> |
|  | *Logit Adjustment* | 0.271 | 0.301 | <u>0.711</u> | <u>0.733</u> | 0.416 | 0.741 |
|  | *PR-entmax* | **0.285**<sup>*</sup> | **0.312**<sup>*</sup> | 0.703 | 0.730 | **0.499**<sup>*</sup> | **0.878**<sup>*</sup> |
| **Yambda-50M** | *Cross-Entropy* | 0.089 | <u>0.118</u> | 0.553 | 0.562 | 0.440 | 0.781 |
|  | *IPW* | 0.085 | 0.106 | **0.687**<sup>*</sup> | <u>0.682</u> | <u>0.696</u> | 0.959 |
|  | *PD* | 0.084 | 0.110 | 0.617 | 0.620 | 0.472 | 0.731 |
|  | *Pop Sampling* | <u>0.092</u> | 0.116 | 0.647 | 0.658 | 0.668 | <u>0.960</u> |
|  | *Logit Adjustment* | 0.089 | 0.113 | 0.626 | 0.637 | 0.550 | 0.878 |
|  | *PR-entmax* | **0.098**<sup>*</sup> | **0.120**<sup>*</sup> | <u>0.671</u> | **0.688**<sup>*</sup> | **0.785**<sup>*</sup> | **0.997**<sup>*</sup> |
| **Yoochoose** | *Cross-Entropy* | <u>0.203</u> | **0.230** | 0.606 | 0.609 | 0.753 | 0.987 |
|  | *IPW* | 0.191 | 0.219 | <u>0.655</u> | 0.680 | 0.832 | **0.999** |
|  | *PD* | 0.188 | 0.216 | 0.645 | 0.655 | 0.768 | 0.977 |
|  | *Pop Sampling* | 0.200 | 0.226 | **0.663**<sup>*</sup> | <u>0.695</u> | <u>0.866</u> | <u>0.998</u> |
|  | *Logit Adjustment* | 0.194 | 0.221 | 0.653 | 0.683 | 0.839 | <u>0.998</u> |
|  | *PR-entmax* | **0.209**<sup>*</sup> | <u>0.228</u> | 0.646 | **0.755**<sup>*</sup> | **0.876**<sup>*</sup> | **0.999**<sup>*</sup> |
| **Zvuk** | *Cross-Entropy* | <u>0.218</u> | **0.256** | 0.550 | 0.539 | 0.163 | 0.462 |
|  | *IPW* | 0.199 | 0.228 | <u>0.655</u> | 0.661 | 0.182 | 0.436 |
|  | *PD* | 0.199 | 0.237 | 0.618 | 0.616 | 0.176 | 0.425 |
|  | *Pop Sampling* | 0.212 | 0.242 | 0.610 | <u>0.668</u> | 0.150 | 0.397 |
|  | *Logit Adjustment* | 0.214 | <u>0.250</u> | 0.637 | 0.657 | <u>0.205</u> | <u>0.559</u> |
|  | *PR-entmax* | **0.222**<sup>*</sup> | 0.246 | **0.686**<sup>*</sup> | **0.806**<sup>*</sup> | **0.216**<sup>*</sup> | **0.579**<sup>*</sup> |



Below we present the trade-off between NDCG@10 and Novelty@10 across different loss functions and hyperparameter configurations for all datasets.

**Figure 1:** Trade-off between NDCG@10 and Novelty@10 for **Tmall** dataset.  
![Tmall dataset](images/tmall_absolute_values.png)

**Figure 2:** Trade-off between NDCG@10 and Novelty@10 for **Yambda-50M** dataset.  
![Yambda-50M dataset](images/yambda_absolute_values.png)

**Figure 3:** Trade-off between NDCG@10 and Novelty@10 for **Yoochoose** dataset.  
![Yoochoose dataset](images/yoochoose_absolute_values.png)

**Figure 4:** Trade-off between NDCG@10 and Novelty@10 for **Zvuk** dataset.  
![Zvuk dataset](images/zvuk_absolute_values.png)

## Usage
Install requirements:
```sh
pip install -r requirements.txt
```
Specify environment variables:
```sh
# path to the project
export PATH4SEQ="/your/path"
# path to the split data
export RECSYS_DATA_PATH="/your/path"
```
Example of run via command line:
```sh
cd runs
# PR-Entmax on Tmall dataset
python dl.py datasets_info=Tmall dataset_params.num_negatives=50000 model_params.num_blocks=2 model_params.num_heads=4 model_params.dropout_rate=0.1 model_params.hidden_units=256 type_loss=entmax temp=1.71 gamma=0.05 seqrec_module.type_correction=logit

# Entmax on Tmall dataset
python dl.py datasets_info=Tmall dataset_params.num_negatives=50000 model_params.num_blocks=2 model_params.num_heads=4 model_params.dropout_rate=0.1 model_params.hidden_units=256 type_loss=entmax temp=1.71

# Logit Adjustment on Tmall dataset
python dl.py datasets_info=Tmall dataset_params.num_negatives=50000 model_params.num_blocks=3 model_params.num_heads=4 model_params.dropout_rate=0.1 model_params.hidden_units=256 type_loss=ce temp=1 seqrec_module.type_correction=logit gamma=0.4

# IPW on Tmall dataset
python dl.py datasets_info=Tmall dataset_params.num_negatives=50000 model_params.num_blocks=3 model_params.num_heads=4 model_params.dropout_rate=0.1 model_params.hidden_units=256 type_loss=ce temp=1 seqrec_module.type_correction=ipw gamma=0.75

# Pop Sampling on Tmall dataset
python dl.py datasets_info=Tmall dataset_params.num_negatives=50000 model_params.num_blocks=3 model_params.num_heads=4 model_params.dropout_rate=0.1 model_params.hidden_units=256 type_loss=ce temp=1 dataset_params.gamma=0.9

# PD on Tmall dataset
python dl.py datasets_info=Tmall dataset_params.num_negatives=50000 model_params.num_blocks=3 model_params.num_heads=4 model_params.dropout_rate=0.1 model_params.hidden_units=256 type_loss=ce temp=1 seqrec_module.type_correction=pd gamma=0.15

# Cross-Entropy on Tmall dataset
python dl.py datasets_info=Tmall dataset_params.num_negatives=50000 model_params.num_blocks=3 model_params.num_heads=4 model_params.dropout_rate=0.1 model_params.hidden_units=256 type_loss=ce temp=1

#Parameter tuning for Tmall with ce loss
python optuna_rec.py datasets_info=Tmall type_loss=ce --multirun

#Parameter tuning for Tmall with entmax loss
python optuna_rec.py datasets_info=Tmall type_loss=entmax --multirun

```
