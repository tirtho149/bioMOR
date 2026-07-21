"""
5-fold CV P-NET — MULTI-CLASS, 3-MODALITY variant (TCGA BRCA PAM50).

Same architecture / loss / multiclass plumbing as
crossvalidation_average_reg_10_tanh_multiclass.py, but ingests a THIRD omics
modality (gene expression) via BRCAMulticlass3ModDataReader, interlacing the
features per gene as [mut, cnv, expr] -> (n_patients, n_genes * 3).

The new data type 'brca_multiclass_3mod' is registered onto data.data_access.Data
here (a small import-time shim) so the original data_access.py is left UNTOUCHED.
"""
from model.builders.prostate_models_multiclass import build_pnet2_multiclass

# -----------------------------------------------------------------------------
# Register the 3-modality reader WITHOUT editing the original data_access.py.
# -----------------------------------------------------------------------------
import data.data_access as _data_access

_DataInit = _data_access.Data.__init__


def _patched_data_init(self, id, type, params, test_size=0.3, stratify=True):
    if type == 'brca_multiclass_3mod':
        self.test_size = test_size
        self.stratify = stratify
        self.data_type = type
        self.data_params = params
        from data.brca.data_reader_multiclass_3mod import BRCAMulticlass3ModDataReader
        self.data_reader = BRCAMulticlass3ModDataReader(**params)
    else:
        _DataInit(self, id, type, params, test_size=test_size, stratify=stratify)


if getattr(_data_access.Data.__init__, "_brca3mod_patched", False) is False:
    _patched_data_init._brca3mod_patched = True
    _data_access.Data.__init__ = _patched_data_init

task = 'classification'

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
NUM_CLASSES = 5

data_base = {
    'id': 'BRCA_PAM50_3MOD',
    'type': 'brca_multiclass_3mod',
    'params': {
        'data_dir': '/lustre/hdd/LAS/weile-lab/howlader/Graph_Transformer/data_tcga/brca',
        'mutation_filename': 'mutation_data.csv',
        'cnv_filename': 'cnv_data.csv',
        'expression_filename': 'expression_data.csv',
        'labels_filename': 'patient_labels_pam50.csv',
        'num_classes': NUM_CLASSES,
        'selected_genes_filename':
            '/lustre/hdd/LAS/weile-lab/howlader/GraphPath/p_net_data/tcga_prostate_expressed_genes_and_cancer_genes.csv',
        'val_size': 10 / 90,
        'test_size': 0.1,
        'random_state': 42,
        'zscore_cnv': True,
        # Expression is continuous; z-score it on the train split like CNV.
        'zscore_expression': True,
    },
}
data = [data_base]

# -----------------------------------------------------------------------------
# Architecture — same depth as the 2-modality multiclass param file
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
            epoch=2,
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

# 5-fold stratified CV (matches the other 3-modality baselines)
pipeline = {'type': 'crossvalidation', 'params': {'n_splits': 2, 'save_train': True}}
