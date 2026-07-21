from model.builders.prostate_models import build_pnet2

task = 'classification_binary'

# -----------------------------------------------------------------------------
# Data config: BRCA / pancancer-style two-modality (mutation + CNV) reader.
# Matches the single-split file `onsplit_average_reg_10_tanh_large_testing.py`
# but the CV pipeline will internally redo train/val/test by k-fold.
# -----------------------------------------------------------------------------
data_base = {'id': 'PAN_META_PRI', 'type': 'brca',
             'params': {
                 'data_dir': '/lustre/hdd/LAS/weile-lab/howlader/Graph_Transformer/data_tcga/pan_meta_pri',
                 'labels_filename': 'patient_labels.csv',
                 'selected_genes_filename': '/lustre/hdd/LAS/weile-lab/howlader/GraphPath/p_net_data/tcga_prostate_expressed_genes_and_cancer_genes.csv',
                 'val_size': 0.10,
                 'test_size': 0.2,
                 'random_state': 42,
                 'zscore_cnv': True,
             }
             }
data = [data_base]

# -----------------------------------------------------------------------------
# Architecture: FIVE pathway hidden layers (original P-NET depth, deep Reactome).
# -----------------------------------------------------------------------------
n_hidden_layers = 5
base_dropout = 0.2
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
                'optimizer': 'Adam',
                'activation': 'tanh',
                'data_params': data_base,
                'add_unk_genes': False,
                'shuffle_genes': False,
                'kernel_initializer': 'lecun_uniform',
                'n_hidden_layers': n_hidden_layers,
                'attention': False,
                'dropout_testing': False
            },
            'fitting_params': dict(samples_per_epoch=10,
                                   # select_best_model=True + monitor='val_o6_f1' caused
                                   # 8/10 folds to collapse on pan_survival_5yr: epoch-2
                                   # checkpoint with spurious val_o6_f1=11.5 was kept and
                                   # never beaten, so the saved weights predicted ~0.5 for
                                   # every patient. Same failure mode as the multiclass
                                   # gotcha — disable both so training uses final-epoch weights.
                                   select_best_model=False,
                                   monitor='val_o6_f1',
                                   verbose=2,
                                   epoch=200,
                                   shuffle=True,
                                   batch_size=16,
                                   save_name='pnet',
                                   debug=False,
                                   save_gradient=False,
                                   class_weight='auto',
                                   n_outputs=n_hidden_layers + 1,
                                   prediction_output='average',
                                   early_stop=False,
                                   reduce_lr=False,
                                   reduce_lr_after_nepochs=dict(drop=0.25, epochs_drop=50),
                                   lr=1e-4,
                                   max_f1=True
                                   ),
            'feature_importance': None
        },
}
features = {}
models = [nn_pathway]

# 5-fold stratified CV. Per fold: ~72% train / ~8% val / ~20% test.
pipeline = {'type': 'crossvalidation', 'params': {'n_splits': 5, 'save_train': True}}
