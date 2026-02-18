# PR-Entmax on Tmall dataset
python dl.py datasets_info=Tmall dataset_params.num_negatives=50000 model_params.num_blocks=2 model_params.num_heads=4 model_params.dropout_rate=0.1 model_params.hidden_units=256 type_loss=entmax temp=1.71 adj=0.05 seqrec_module.type_adj=logit random_state=17,42,52 --multirun

# Entmax on Tmall dataset
python dl.py datasets_info=Tmall dataset_params.num_negatives=50000 model_params.num_blocks=2 model_params.num_heads=4 model_params.dropout_rate=0.1 model_params.hidden_units=256 type_loss=entmax temp=1.71 random_state=17,42,52 --multirun

# Logit Adjustment on Tmall dataset
python dl.py datasets_info=Tmall dataset_params.num_negatives=50000 model_params.num_blocks=3 model_params.num_heads=4 model_params.dropout_rate=0.1 model_params.hidden_units=256 type_loss=ce temp=1 seqrec_module.type_adj=logit adj=0.4 random_state=17,42,52 --multirun

# Margin Adjustment on Tmall dataset
python dl.py datasets_info=Tmall dataset_params.num_negatives=50000 model_params.num_blocks=3 model_params.num_heads=4 model_params.dropout_rate=0.1 model_params.hidden_units=256 type_loss=ce temp=1 seqrec_module.type_adj=margin adj=0.01 random_state=17,42,52 --multirun

# Cross-Entropy on Tmall dataset
python dl.py datasets_info=Tmall dataset_params.num_negatives=50000 model_params.num_blocks=3 model_params.num_heads=4 model_params.dropout_rate=0.1 model_params.hidden_units=256 type_loss=ce temp=1 random_state=17,42,52 --multirun


# PR-Entmax on Zvuk dataset
python dl.py datasets_info=Zvuk model_params.num_blocks=1 model_params.num_heads=4 model_params.dropout_rate=0.4 model_params.hidden_units=224 type_loss=entmax temp=26 adj=1 seqrec_module.type_adj=logit random_state=17,42,52 --multirun

# Entmax on Zvuk dataset
python dl.py datasets_info=Zvuk model_params.num_blocks=1 model_params.num_heads=4 model_params.dropout_rate=0.4 model_params.hidden_units=224 type_loss=entmax temp=26 random_state=17,42,52 --multirun

# Logit Adjustment on Zvuk dataset
python dl.py datasets_info=Zvuk model_params.num_blocks=1 model_params.num_heads=2 model_params.dropout_rate=0.5 model_params.hidden_units=192 type_loss=ce temp=1 seqrec_module.type_adj=logit adj=0.8 random_state=17,42,52 --multirun

# Margin Adjustment on Zvuk dataset
python dl.py datasets_info=Zvuk model_params.num_blocks=1 model_params.num_heads=2 model_params.dropout_rate=0.5 model_params.hidden_units=192 type_loss=ce temp=1 seqrec_module.type_adj=margin adj=10 random_state=17,42,52 --multirun

# Cross-Entropy on Zvuk dataset
python dl.py datasets_info=Zvuk model_params.num_blocks=1 model_params.num_heads=2 model_params.dropout_rate=0.5 model_params.hidden_units=192 type_loss=ce temp=1 random_state=17,42,52 --multirun


# PR-Entmax on Yambda dataset
python dl.py datasets_info=Yambda model_params.num_blocks=1 model_params.num_heads=1 model_params.dropout_rate=0.2 model_params.hidden_units=256 type_loss=entmax temp=42 adj=0.9 seqrec_module.type_adj=logit random_state=17,42,52 --multirun

# Entmax on Yambda dataset
python dl.py datasets_info=Yambda model_params.num_blocks=1 model_params.num_heads=1 model_params.dropout_rate=0.2 model_params.hidden_units=256 type_loss=entmax temp=42 random_state=17,42,52 --multirun

# Logit Adjustment on Yambda dataset
python dl.py datasets_info=Yambda model_params.num_blocks=2 model_params.num_heads=2 model_params.dropout_rate=0.2 model_params.hidden_units=256 type_loss=ce temp=1 seqrec_module.type_adj=logit adj=0.6 random_state=17,42,52 --multirun

# Margin Adjustment on Yambda dataset
python dl.py datasets_info=Yambda model_params.num_blocks=2 model_params.num_heads=2 model_params.dropout_rate=0.2 model_params.hidden_units=256 type_loss=ce temp=1 seqrec_module.type_adj=margin adj=5 random_state=17,42,52 --multirun

# Cross-Entropy on Yambda dataset
python dl.py datasets_info=Yambda model_params.num_blocks=2 model_params.num_heads=2 model_params.dropout_rate=0.2 model_params.hidden_units=256 type_loss=ce temp=1 random_state=17,42,52 --multirun


# PR-Entmax on Yoochoose dataset
python dl.py datasets_info=Yoochoose model_params.num_blocks=3 model_params.num_heads=1 model_params.dropout_rate=0.3 model_params.hidden_units=224 type_loss=entmax temp=8.2 adj=0.5 seqrec_module.type_adj=logit random_state=17,42,52 --multirun

# Entmax on Yoochoose dataset
python dl.py datasets_info=Yoochoose model_params.num_blocks=3 model_params.num_heads=1 model_params.dropout_rate=0.3 model_params.hidden_units=224 type_loss=entmax temp=8.2 random_state=17,42,52 --multirun

# Logit Adjustment on Yoochoose dataset
python dl.py datasets_info=Yoochoose model_params.num_blocks=2 model_params.num_heads=1 model_params.dropout_rate=0.3 model_params.hidden_units=128 type_loss=ce temp=1 seqrec_module.type_adj=logit adj=0.9 random_state=17,42,52 --multirun

# Margin Adjustment on Yoochoose dataset
python dl.py datasets_info=Yoochoose model_params.num_blocks=2 model_params.num_heads=1 model_params.dropout_rate=0.3 model_params.hidden_units=128 type_loss=ce temp=1 seqrec_module.type_adj=margin adj=15 random_state=17,42,52 --multirun

# Cross-Entropy on Yoochoose dataset
python dl.py datasets_info=Yoochoose model_params.num_blocks=2 model_params.num_heads=1 model_params.dropout_rate=0.3 model_params.hidden_units=128 type_loss=ce temp=1 random_state=17,42,52 --multirun
