"""
10-fold CV P-NET — KIRC molecular-subtype (4-class).

Uses:
    - data type 'brca_multiclass' (BRCAMulticlassDataReader works for any
      multi-class 2-modality dataset)
    - build_pnet2_multiclass model builder (softmax outputs)

Same architecture and fix-pattern as the pan_brca_molsubtype params file
(select_best_model=False + early_stop=False so training runs the full
200 epochs and uses final-epoch weights).
"""
from model.builders.prostate_models_multiclass import build_pnet2_multiclass

task = 'classification'

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
NUM_CLASSES = 4

data_base = {
    'id': 'KIRC_MOLSUBTYPE',
    'type': 'brca_multiclass',
    'params': {
        'data_dir': '/lustre/hdd/LAS/weile-lab/howlader/Graph_Transformer/data_tcga/kirc_molsubtype',
        'labels_filename': 'patient_labels.csv',
        'num_classes': NUM_CLASSES,
        'selected_genes_filename':
            '/lustre/hdd/LAS/weile-lab/howlader/GraphPath/p_net_data/tcga_prostate_expressed_genes_and_cancer_genes.csv',
        'val_size': 10 / 90,
        'test_size': 0.1,
        'random_state': 42,
        'zscore_cnv': True,
    },
}
data = [data_base]

# -----------------------------------------------------------------------------
# Architecture — same depth as the binary param file
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
    'params': {
        'build_fn': build_pnet2_multiclass,
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
            'dropout_testing': False,
            'num_classes': NUM_CLASSES,
            'loss': 'sparse_categorical_crossentropy',
        },
        'fitting_params': dict(
            samples_per_epoch=10,
            select_best_model=False,
            monitor='val_loss',
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
            max_f1=False,
        ),
        'feature_importance': None,
    },
}
features = {}
models = [nn_pathway]

# 10-fold stratified CV
pipeline = {'type': 'crossvalidation', 'params': {'n_splits': 10, 'save_train': True}}
