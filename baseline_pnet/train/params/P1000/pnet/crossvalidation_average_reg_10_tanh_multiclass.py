"""
10-fold CV P-NET — MULTI-CLASS variant.

Uses:
    - data type 'brca_multiclass' (BRCAMulticlassDataReader)
    - build_pnet2_multiclass model builder (softmax outputs)
    - task = 'classification' → evalualte_classification_multiclass
      (macro-OvR AUC/AUPR, macro F1)

Switch dataset by editing data_base['id'], data_base['params']['data_dir'],
and data_base['params']['num_classes'].
"""
from model.builders.prostate_models_multiclass import build_pnet2_multiclass

task = 'classification'

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
NUM_CLASSES = 5

data_base = {
    'id': 'PAN_BRCA_MOLSUBTYPE',
    'type': 'brca_multiclass',
    'params': {
        'data_dir': '/lustre/hdd/LAS/weile-lab/howlader/Graph_Transformer/data_tcga/pan_brca_molsubtype',
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
            # Multiclass loss — also gates predict/predict_proba in nn.py
            # to return argmax/full softmax instead of binary thresholding.
            'loss': 'sparse_categorical_crossentropy',
        },
        'fitting_params': dict(
            samples_per_epoch=10,
            # Multiclass: select_best_model + early_stop fire prematurely on
            # this codebase's monitors (val_o6_accuracy is stuck; FixedEarly-
            # Stopping mishandles val_loss under Keras-3 multi-output logs).
            # Disable both and use the final epoch's weights.
            select_best_model=False,
            # Multiclass: deep heads (o2..o6) stay near-uniform softmax under
            # sparse_categorical loss, and val_o1_accuracy peaks at epoch 1
            # — both stick the ModelCheckpoint / EarlyStopping to ~epoch 1.
            # val_loss aggregates all 6 weighted head losses and DOES improve
            # over training (o1's loss × weight_2 alone moves it), so this
            # lets training run longer and saves a later, sharper-softmax
            # checkpoint.
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
            max_f1=False,  # binary-only threshold search, off for multi-class
        ),
        'feature_importance': None,
    },
}
features = {}
models = [nn_pathway]

# 10-fold stratified CV
pipeline = {'type': 'crossvalidation', 'params': {'n_splits': 10, 'save_train': True}}
