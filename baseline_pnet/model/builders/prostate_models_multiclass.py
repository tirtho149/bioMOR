"""
Multi-class variant of build_pnet2.

Identical architecture (Diagonal h0 → SparseTF h1..hN), but decision heads
emit `Dense(num_classes, activation='softmax')` instead of
`Dense(1, activation='sigmoid')`. Compiled with `sparse_categorical_crossentropy`
so labels can stay as `(N,)` int arrays.

Usage in a param file:

    from model.builders.prostate_models_multiclass import build_pnet2_multiclass

    nn_pathway = {'type': 'nn', 'id': 'P-net', 'params': {
        'build_fn': build_pnet2_multiclass,
        'model_params': {
            'num_classes': 5,
            ...
        },
        ...
    }}
"""
import logging

import numpy as np
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout, Input, multiply
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from data.data_access import Data
from model.builders.builders_utils import get_layer_maps, shuffle_genes_map
from model.layers_custom import Diagonal, SparseTF


def _print_model(model):
    for l in model.layers:
        try:
            shape = l.output.shape
        except Exception:
            shape = "?"
        logging.info("layer %s : %s", l.name, shape)


def get_pnet_multiclass(inputs, features, genes, n_hidden_layers, direction, activation,
                        num_classes, w_reg, w_reg_outcomes, dropout, sparse, add_unk_genes,
                        batch_normal, kernel_initializer, use_bias=False, shuffle_genes=False,
                        attention=False, dropout_testing=False, non_neg=False,
                        sparse_first_layer=True):
    """Same backbone as get_pnet, but decision outputs are
    Dense(num_classes, softmax) so the model can do K-class classification."""
    feature_names = {}
    n_features = len(features)
    n_genes = len(genes)

    if not isinstance(w_reg, list):
        w_reg = [w_reg] * 10
    if not isinstance(w_reg_outcomes, list):
        w_reg_outcomes = [w_reg_outcomes] * 10
    if not isinstance(dropout, list):
        dropout = [dropout] * 10

    w_reg0 = w_reg[0]
    w_reg_outcome1 = w_reg_outcomes[1]
    reg_l = l2
    constraints = {}
    if non_neg:
        from tensorflow.keras.constraints import non_neg as _nonneg
        constraints = {'kernel_constraint': _nonneg()}

    if sparse:
        if shuffle_genes == 'all':
            ones_ratio = float(n_features) / np.prod([n_genes, n_features])
            mapp = np.random.choice([0, 1], size=[n_features, n_genes], p=[1 - ones_ratio, ones_ratio])
            layer1 = SparseTF(n_genes, mapp, activation=activation, W_regularizer=reg_l(w_reg0),
                              name='h0', kernel_initializer=kernel_initializer, use_bias=use_bias,
                              **constraints)
        else:
            layer1 = Diagonal(n_genes, input_shape=(n_features,), activation=activation,
                              W_regularizer=l2(w_reg0), use_bias=use_bias, name='h0',
                              kernel_initializer=kernel_initializer, **constraints)
    else:
        if sparse_first_layer:
            layer1 = Diagonal(n_genes, input_shape=(n_features,), activation=activation,
                              W_regularizer=l2(w_reg0), use_bias=use_bias, name='h0',
                              kernel_initializer=kernel_initializer, **constraints)
        else:
            layer1 = Dense(n_genes, input_shape=(n_features,), activation=activation,
                           kernel_regularizer=l2(w_reg0), use_bias=use_bias, name='h0',
                           kernel_initializer=kernel_initializer)

    outcome = layer1(inputs)
    if attention:
        attention_probs = Diagonal(n_genes, input_shape=(n_features,), activation='sigmoid',
                                   W_regularizer=l2(w_reg0), name='attention0')(inputs)
        outcome = multiply([outcome, attention_probs], name='attention_mul')

    decision_outcomes = []

    # First decision head — direct from input layer
    decision_outcome = Dense(num_classes, activation='linear', name='o_linear1',
                             kernel_regularizer=reg_l(w_reg_outcome1 / 2.))(outcome)
    drop2 = Dropout(dropout[0], name='dropout_0')
    outcome = drop2(outcome, training=dropout_testing)
    if batch_normal:
        decision_outcome = BatchNormalization()(decision_outcome)
    decision_outcome = Activation(activation='softmax', name='o1')(decision_outcome)
    decision_outcomes.append(decision_outcome)

    if n_hidden_layers > 0:
        maps = get_layer_maps(genes, n_hidden_layers, direction, add_unk_genes)
        w_regs = w_reg[1:]
        w_reg_outcomes_t = w_reg_outcomes[1:]
        dropouts = dropout[1:]
        for i, mapp in enumerate(maps[0:-1]):
            w_reg_i = w_regs[i]
            w_reg_outcome = w_reg_outcomes_t[i]
            drop_i = dropouts[1] if len(dropouts) > 1 else dropouts[0]
            names = mapp.index
            mapp = mapp.values
            if shuffle_genes in ['all', 'pathways']:
                mapp = shuffle_genes_map(mapp)
            n_genes, n_pathways = mapp.shape
            logging.info('n_genes=%d n_pathways=%d', n_genes, n_pathways)
            layer_name = 'h{}'.format(i + 1)
            if sparse:
                hidden_layer = SparseTF(n_pathways, mapp, activation=activation,
                                        W_regularizer=reg_l(w_reg_i), name=layer_name,
                                        kernel_initializer=kernel_initializer,
                                        use_bias=use_bias, **constraints)
            else:
                hidden_layer = Dense(n_pathways, activation=activation,
                                     kernel_regularizer=reg_l(w_reg_i), name=layer_name,
                                     kernel_initializer=kernel_initializer, **constraints)

            outcome = hidden_layer(outcome)

            if attention:
                attention_probs = Dense(n_pathways, activation='sigmoid',
                                        name='attention{}'.format(i + 1),
                                        kernel_regularizer=l2(w_reg_i))(outcome)
                outcome = multiply([outcome, attention_probs],
                                   name='attention_mul{}'.format(i + 1))

            decision_outcome = Dense(num_classes, activation='linear',
                                     name='o_linear{}'.format(i + 2),
                                     kernel_regularizer=reg_l(w_reg_outcome))(outcome)
            if batch_normal:
                decision_outcome = BatchNormalization()(decision_outcome)
            decision_outcome = Activation(activation='softmax',
                                          name='o{}'.format(i + 2))(decision_outcome)
            decision_outcomes.append(decision_outcome)
            drop = Dropout(drop_i, name='dropout_{}'.format(i + 1))
            outcome = drop(outcome, training=dropout_testing)

            feature_names['h{}'.format(i)] = names

        i = len(maps)
        feature_names['h{}'.format(i - 1)] = maps[-1].index

    return outcome, decision_outcomes, feature_names


def build_pnet2_multiclass(optimizer, w_reg, w_reg_outcomes, num_classes,
                           add_unk_genes=True, sparse=True, loss_weights=1.0,
                           dropout=0.5, use_bias=False, activation='tanh',
                           data_params=None, n_hidden_layers=1,
                           direction='root_to_leaf', batch_normal=False,
                           kernel_initializer='glorot_uniform', shuffle_genes=False,
                           attention=False, dropout_testing=False, non_neg=False,
                           repeated_outcomes=True, sparse_first_layer=True,
                           # ignored kwargs (kept for param-file compatibility with binary builder):
                           loss=None, **_unused):
    """Multi-class P-NET. Returns (model, feature_names).

    Compiled with sparse_categorical_crossentropy and accuracy metrics. Labels
    should be int arrays of shape (N,) with values in {0, ..., num_classes-1}.
    """
    logging.info('build_pnet2_multiclass  num_classes=%s', num_classes)
    data = Data(**data_params)
    x, y, info, cols = data.get_data()
    features = cols

    n_features = x.shape[1]
    if hasattr(cols, 'levels'):
        genes = cols.levels[0]
    else:
        genes = cols

    ins = Input(shape=(n_features,), dtype='float32', name='inputs')

    outcome, decision_outcomes, feature_n = get_pnet_multiclass(
        ins, features=features, genes=genes,
        n_hidden_layers=n_hidden_layers, direction=direction,
        activation=activation, num_classes=num_classes,
        w_reg=w_reg, w_reg_outcomes=w_reg_outcomes, dropout=dropout,
        sparse=sparse, add_unk_genes=add_unk_genes,
        batch_normal=batch_normal, sparse_first_layer=sparse_first_layer,
        use_bias=use_bias, kernel_initializer=kernel_initializer,
        shuffle_genes=shuffle_genes, attention=attention,
        dropout_testing=dropout_testing, non_neg=non_neg,
    )

    feature_names = feature_n
    feature_names['inputs'] = cols

    if repeated_outcomes:
        outcome = decision_outcomes
    else:
        outcome = decision_outcomes[-1]

    model = Model(inputs=[ins], outputs=outcome)

    n_outputs = len(outcome) if isinstance(outcome, list) else 1
    if not isinstance(loss_weights, list):
        loss_weights = [loss_weights] * n_outputs

    model.compile(
        optimizer=optimizer,
        loss=['sparse_categorical_crossentropy'] * n_outputs,
        metrics=[['accuracy']] * n_outputs,
        loss_weights=loss_weights,
    )
    logging.info('multiclass model compiled; n_outputs=%d, num_classes=%d',
                 n_outputs, num_classes)
    _print_model(model)
    return model, feature_names
