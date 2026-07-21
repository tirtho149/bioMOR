from model.builders.prostate_models import build_pnet2

task = 'classification_binary'

# -----------------------------------------------------------------------------
# Data config: BRCA / pancancer-style two-modality (mutation + CNV) reader.
# Edit `data_dir` to point at the directory containing mutation_data.csv,
# cnv_data.csv, and your labels CSV.
# -----------------------------------------------------------------------------
data_base = {'id': 'BLCA', 'type': 'brca',
             'params': {
                 'data_dir': '/lustre/hdd/LAS/weile-lab/howlader/Graph_Transformer/data/blca',
                 'labels_filename': 'patient_labels.csv',
                 'selected_genes_filename': '/lustre/hdd/LAS/weile-lab/howlader/GraphPath/p_net_data/tcga_prostate_expressed_genes_and_cancer_genes.csv',
                 'val_size': 10 / 90,     # 10% val (of 90% train_val) → 10% of total
                 'test_size': 0.1,        # 10% test of total
                 'random_state': 42,
                 'zscore_cnv': True,
             }
             }
data = [data_base]

# -----------------------------------------------------------------------------
# Architecture: FIVE pathway hidden layers (original P-NET depth).
#
# Using the authentic P-NET Reactome hierarchy under
# `_database/pathways/Reactome/` (ReactomePathways.{txt,gmt} +
# ReactomePathwaysRelation.txt). With n_hidden_layers=5 the architecture is:
#     features → Diagonal h0 (genes) → SparseTF h1..h5 (pathway layers) → outputs
# n_outputs = n_hidden_layers + 1 = 6 → final decision head is `o6`.
# -----------------------------------------------------------------------------
n_hidden_layers = 5
base_dropout = 0.2                         # was 0.5; aligned with SKILLS.md
wregs = [0.001] * 7
loss_weights = [2, 7, 20, 54, 148, 400]
wreg_outcomes = [0.01] * 6
pre = {'type': None}

nn_pathway = {
    'type': 'nn',
    'id': 'P-net',
    'params':
        {
            'build_fn': build_pnet2,
            'model_params': {
                'use_bias': True,
                'w_reg': wregs,
                'w_reg_outcomes': wreg_outcomes,
                'dropout': [base_dropout] + [0.1] * (n_hidden_layers + 1),
                'loss_weights': loss_weights,
                # Pass a string ('Adam' / 'AdamW') OR instantiate a tf.keras
                # optimizer directly for finer control. SKILLS.md prefers
                # AdamW(weight_decay=5e-4). To switch, change to:
                #     'optimizer': 'AdamW',     # Keras 3 accepts the string
                # or pass an instance, e.g.:
                #     import tensorflow as tf
                #     'optimizer': tf.keras.optimizers.AdamW(
                #         learning_rate=1e-4, weight_decay=5e-4)
                'optimizer': 'Adam',
                'activation': 'tanh',
                'data_params': data_base,
                'add_unk_genes': False,
                'shuffle_genes': False,
                'kernel_initializer': 'lecun_uniform',
                'n_hidden_layers': n_hidden_layers,
                'attention': False,
                'dropout_testing': False  # keep dropout in testing phase, useful for bayesian inference

            }, 'fitting_params': dict(samples_per_epoch=10,
                                      select_best_model=True,    # use best-val
                                      # `monitor` watches the final decision
                                      # head; with n_hidden_layers=5 that's o6.
                                      monitor='val_o6_f1',
                                      verbose=2,
                                      epoch=200,                  # SKILLS.md: 200
                                      shuffle=True,
                                      batch_size=16,              # SKILLS.md: 16
                                      save_name='pnet',
                                      debug=False,
                                      save_gradient=False,
                                      class_weight='auto',
                                      n_outputs=n_hidden_layers + 1,
                                      prediction_output='average',
                                      early_stop=True,            # SKILLS.md: on
                                      reduce_lr=False,
                                      reduce_lr_after_nepochs=dict(drop=0.25, epochs_drop=50),
                                      lr=1e-4,                    # SKILLS.md: 1e-4
                                      max_f1=True
                                      ),
            # 'deepexplain_*' is not supported in the TF2 port (DeepExplain
            # is a TF1-graph-mode library). Set to None to skip per-feature
            # interpretation — y_score and metrics still get exported.
            'feature_importance': None
        },
}
features = {}
models = [nn_pathway]

# (The original param file appended a secondary logistic-regression model with
# hardcoded class weights tuned for the prostate dataset. Dropped here because
# we only need the P-NET baseline. If you want logistic-regression as an
# additional sanity-check baseline, append a config like:
#
#   logistic = {'type': 'sgd', 'id': 'Logistic Regression',
#               'params': {'loss': 'log_loss', 'penalty': 'l2',
#                          'alpha': 0.01, 'class_weight': 'balanced'}}
#   models.append(logistic)
# )

pipeline = {'type': 'one_split', 'params': {'save_train': True, 'eval_dataset': 'test'}}
