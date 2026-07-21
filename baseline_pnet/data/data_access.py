import logging
import numpy as np

# NOTE: The original prostate-paper data reader has been removed in this slim
# variant of P-NET. To use this code with a new dataset (e.g. BRCA / pancancer
# from the Graph_Transformer pipeline), implement a data reader class with the
# same interface as the original ProstateDataPaper:
#
#   class MyDataReader:
#       def __init__(self, **params): ...
#       def get_train_validate_test(self):
#           return (x_train, x_validate, x_test,
#                   y_train, y_validate, y_test,
#                   info_train, info_validate, info_test,
#                   columns)        # `columns` is the feature names
#       self.x, self.y, self.info, self.columns       # whole-dataset getters
#
# Drop it under `data/<dataset_name>/data_reader.py`, import it here, and add
# the appropriate `if self.data_type == '<dataset_name>':` branch below.


class Data():
    def __init__(self, id, type, params, test_size=0.3, stratify=True):

        self.test_size = test_size
        self.stratify = stratify
        self.data_type = type
        self.data_params = params

        if self.data_type == 'brca':
            from data.brca.data_reader import BRCADataReader
            self.data_reader = BRCADataReader(**params)
        elif self.data_type == 'brca_multiclass':
            from data.brca.data_reader_multiclass import BRCAMulticlassDataReader
            self.data_reader = BRCAMulticlassDataReader(**params)
        # To add another dataset (e.g. pancancer), drop a data_reader.py
        # under data/<name>/ exposing the same interface and add a branch:
        #
        #   elif self.data_type == 'pancancer':
        #       from data.pancancer.data_reader import PancancerDataReader
        #       self.data_reader = PancancerDataReader(**params)
        else:
            logging.error('unsupported data type: {}'.format(self.data_type))
            raise NotImplementedError(
                "No data reader registered for data type '{}'. "
                "See the docstring at the top of data/data_access.py.".format(self.data_type)
            )

    def get_train_validate_test(self):
        return self.data_reader.get_train_validate_test()

    def get_train_test(self):
        x_train, x_validate, x_test, y_train, y_validate, y_test, info_train, info_validate, info_test, columns = self.data_reader.get_train_validate_test()
        # combine training and validation datasets
        x_train = np.concatenate((x_train, x_validate))
        y_train = np.concatenate((y_train, y_validate))
        info_train = list(info_train) + list(info_validate)
        return x_train, x_test, y_train, y_test, info_train, info_test, columns

    def get_data(self):
        x = self.data_reader.x
        y = self.data_reader.y
        info = self.data_reader.info
        columns = self.data_reader.columns
        return x, y, info, columns

    def get_relevant_features(self):
        if hasattr(self.data_reader, 'relevant_features'):
            return self.data_reader.get_relevant_features()
        else:
            return None
