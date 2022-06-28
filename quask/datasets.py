"""
Module dedicated to the generation and retrieval of classical and quantum datasets.

Args:
    the_dataset_register: singleton global instance of DatasetRegister
"""


import openml
import numpy as np


def download_dataset_openml(the_id):
    """
    Download a dataset from OpenML platform given the ID of the dataset.

    Args:
        the_id: ID of the dataset (int)

    Returns:
        tuple (X, y) of numpy ndarray having shapes (d,n) and (n,)
    """
    metadata = openml.datasets.get_dataset(the_id)
    # get dataset
    X, y, _, attribute_names = metadata.get_data(
        dataset_format="array", target=metadata.default_target_attribute
    )
    return X, y


def download_dataset_openml_by_name(name):
    """
    Download a dataset from OpenML platform given the name of the dataset

    Args:
        name: name of the dataset (str)

    Returns:
        tuple (X, y) of numpy ndarray having shapes (d,n) and (n,)
    """
    # get the list of all datasets in OpenML platform
    openml_df = openml.datasets.list_datasets(output_format="dataframe")
    # retrieve the dataset id
    the_id = int(openml_df.query(f'name == "{name}"').iloc[0]["did"])
    return download_dataset_openml_by_name(the_id)


def get_dataset_quantum(the_id):
    """
    This function calls already preprocessed datasets with quantum labels.
    These examples are identified with a specific id.
    The available datasets at the moment are:
        - Fashion-MNIST with 2 features and encoding E1
        - Fashion-MNIST with 4 features and encoding E2
        - Fashion-MNIST with 8 features and encoding E3

    These datasets can be used to benchmark the performace of our
    classical and quantum kernels to verify the power of data.

    Args
        the_id: parameter able to distinguish between quantum dataset

    Returns:
        tuple (X, y) of numpy ndarray having shapes (d,n) and (n,)
    """
    try:
        import importlib.resources as pkg_resources
    except ImportError:
        # Try backported to PY<37 `importlib_resources`.
        import importlib_resources as pkg_resources

    from . import resources

    if the_id == 0:
        X = np.loadtxt(
            pkg_resources.open_text(resources, "X_Q_Fashion-MNIST_720_2_E1"),
            delimiter=" ",
        )
        y = np.loadtxt(
            pkg_resources.open_text(resources, "y_Q_Fashion-MNIST_720_2_E1"),
            delimiter=" ",
        )
    elif the_id == 1:
        X = np.loadtxt(
            pkg_resources.open_text(resources, "X_Q_Fashion-MNIST_720_4_E2"),
            delimiter=" ",
        )
        y = np.loadtxt(
            pkg_resources.open_text(resources, "y_Q_Fashion-MNIST_720_4_E2"),
            delimiter=" ",
        )
    elif the_id == 2:
        X = np.loadtxt(
            pkg_resources.open_text(resources, "X_Q_Fashion-MNIST_720_8_E3"),
            delimiter=" ",
        )
        y = np.loadtxt(
            pkg_resources.open_text(resources, "y_Q_Fashion-MNIST_720_8_E3"),
            delimiter=" ",
        )

    return X, y


class DatasetRegister:
    """
    List of datasets available in this module. The object is iterable.
    """

    def __init__(self):
        """
        Init method.

        Returns:
            None
        """
        self.datasets = []
        self.current = 0

    def register(self, dataset_name, dataset_type, information_nature, get_dataset):
        """
        Register a new dataset.

        Args:
            dataset_name: name of the dataset
            dataset_type: 'regression' or 'classification'
            information_nature: 'classical' or 'quantum'
            get_dataset: function pointer to a zero-parameter function returning (X, y)

        Returns:
            None
        """
        assert dataset_type in ["regression", "classification"]
        assert information_nature in ["classical", "quantum"]
        self.datasets.append(
            {
                "name": dataset_name,
                "type": dataset_type,
                "information": information_nature,
                "get_dataset": get_dataset,
            }
        )

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= len(self.datasets):
            raise StopIteration
        self.current += 1
        return self.datasets[self.current - 1]

    def __len__(self):
        return len(self.datasets)


the_dataset_register = DatasetRegister()
the_dataset_register.register(
    "iris", "classification", "classical", lambda: download_dataset_openml(61)
)
the_dataset_register.register(
    "Fashion-MNIST",
    "classification",
    "classical",
    lambda: download_dataset_openml(40996),
)
the_dataset_register.register(
    "liver-disorders", "regression", "classical", lambda: download_dataset_openml(8)
)
the_dataset_register.register(
    "delta_elevators", "regression", "classical", lambda: download_dataset_openml(198)
)
the_dataset_register.register(
    "Q_Fashion-MNIST_2_E1", "regression", "quantum", lambda: get_dataset_quantum(0)
)
the_dataset_register.register(
    "Q_Fashion-MNIST_4_E2", "regression", "quantum", lambda: get_dataset_quantum(1)
)
the_dataset_register.register(
    "Q_Fashion-MNIST_8_E3", "regression", "quantum", lambda: get_dataset_quantum(2)
)


# class Data:
#     def __init__(self,
#                  data: Optional[float] = None,
#                  n_datapoints: Optional[int] = None,
#                  n_features: Optional[int] = None,
#                  features: Optional[list] = None,
#                  filename: Optional[str] = None,
#                  path: Optional[str] = None,
#                  ):
#         """
#         init
#
#         This function initializes the class with the following attributes.
#
#         n_features: int number of features of each datapoint
#         n_datapoints: int number of datapoints in the dataset
#         features: list of strings labelling each feature
#         datatype: string 'train', 'test', or 'validation'
#         origin: string 'bkg' or 'sig' to identify the initial dataset containing
#                 only background or signal events only
#         path: string indicating the file path
#
#         """
#         self.data = data
#         self.n_datapoints = n_datapoints
#         self.n_features = n_features
#         self.features = features
#         self.filename = filename
#         self.path = path
#
#     def data_load(self, datatype):
#         """
#         load background/signal
#
#         This function loads the background and signal dataset.
#
#         """
#         # fetching the dataset from CERNBox drive
#
#         url = 'https://cernbox.cern.ch/index.php/s/WAKFsaxC9aSW59R?path=%2Finput_ae'
#         if datatype == 'train':
#             name = 'x_data_minmax_7.20e+05_train.npy'
#         elif datatype == 'test':
#             name = 'x_data_minmax_7.20e+05_test.npy'
#         elif datatype == 'valid':
#             name = 'x_data_minmax_7.20e+05_valid.npy'
#
#         r = requests.get(url + name).content
#         with open(self.filename, 'wb') as self.data:
#             self.data.write(r)
#
#         print(self.data[:10])
#
#         # self.data = np.load(f'{self.path+self.filename}')
#         self.n_datapoints = self.data.shape[0]
#         self.n_features = self.data.shape[1]
#         return self
#
#     def label_load(self, datatype):
#         """
#         load background/signal
#
#         This function loads the background and signal dataset.
#
#         """
#         # fetching the background dataset from its path
#
#         url = 'https://cernbox.cern.ch/index.php/s/WAKFsaxC9aSW59R?path=%2Finput_ae'
#         if datatype == 'train':
#             name = 'y_data_minmax_7.20e+05_train.npy'
#         elif datatype == 'test':
#             name = 'y_data_minmax_7.20e+05_test.npy'
#         elif datatype == 'valid':
#             name = 'y_data_minmax_7.20e+05_valid.npy'
#
#         r = requests.get(url + name).content
#         with open(self.filename, 'wb') as self.data:
#             self.data.write(r)
#
#         print(self.data[:10])
#
#         # self.data = np.load(f'{self.path+self.filename}')
#         self.n_datapoints = self.data.shape[0]
#         return self
#
#     def get_feature_names(self):
#         """
#         get feature names
#
#         This function allocate the features of the Higgs event under study
#         to the dataset.
#         """
#
#         features = ['jet_1_pt', 'jet_1_eta', 'jet_1_phi', 'jet_1_energy', 'jet_1_b-tag', 'jet_1_px', 'jet_1_py',
#                     'jet_1_pz',
#                     'jet_2_pt', 'jet_2_eta', 'jet_2_phi', 'jet_2_energy', 'jet_2_b-tag', 'jet_2_px', 'jet_2_py',
#                     'jet_2_pz',
#                     'jet_3_pt', 'jet_3_eta', 'jet_3_phi', 'jet_3_energy', 'jet_3_b-tag', 'jet_3_px', 'jet_3_py',
#                     'jet_3_pz',
#                     'jet_4_pt', 'jet_4_eta', 'jet_4_phi', 'jet_4_energy', 'jet_4_b-tag', 'jet_4_px', 'jet_4_py',
#                     'jet_4_pz',
#                     'jet_5_pt', 'jet_5_eta', 'jet_5_phi', 'jet_5_energy', 'jet_5_b-tag', 'jet_5_px', 'jet_5_py',
#                     'jet_5_pz',
#                     'jet_6_pt', 'jet_6_eta', 'jet_6_phi', 'jet_6_energy', 'jet_6_b-tag', 'jet_6_px', 'jet_6_py',
#                     'jet_6_pz',
#                     'jet_7_pt', 'jet_7_eta', 'jet_7_phi', 'jet_7_energy', 'jet_7_b-tag', 'jet_7_px', 'jet_7_py',
#                     'jet_7_pz',
#                     'met_pt', 'met_phi', 'met_px', 'met_py',
#                     'lept_pt', 'lept_eta', 'lept_phi', 'lept_energy', 'lept_px', 'lept_py', 'lept_pz']
#
#         self.features = features
#
#     def compute_auc_roc(self, labels):
#         """
#         compute auc roc
#
#         This function compute the Area Under the Curve (AUC) of the ROC curve for each feature in the dataset.
#
#         self: dataset
#         labels: labels for the given dataset. '0' for background, '1' for signal
#         """
#
#         # divide the dataset in signal and background
#         # x_bkg = []
#         # x_sig = []
#         # for i in range(self.n_datapoints):
#
#         #     if labels[i] == 0:
#         #         x_bkg.append(self.data[i])
#
#         #     elif labels[i] == 1:
#         #         x_sig.append(self.data[i])
#
#         # x_bkg = np.array(x_bkg)
#         # x_sig = np.array(x_sig)
#
#         aucs = []
#         for i in range(self.n_features):
#
#             # Compute ROC curve and ROC area for each class
#             fpr, tpr, _ = roc_curve(labels, self.data[:, i])
#             roc_auc = auc(fpr, tpr)
#
#             # results under 0.5 are still discriminative
#             if roc_auc < 0.5:
#                 roc_auc = 1 - roc_auc
#
#             aucs.append(roc_auc)
#
#         # save the auc scores relatively to their features
#         feature_scores = {feature: score for feature, score in zip(self.features, aucs)}
#         return feature_scores
#
#     def feature_selection(self, feature_scores=dict, n_features=int):
#         """
#         feature selection
#
#         This function selects n_features from the dataset according to their auc scores.
#
#         self: dataset
#         feature_scores: dict with scores of the roc auc for each feature ordered as in the dataset
#         n_features: number of features for the final dataset
#         """
#
#         # save only the values of the scores
#         scores = []
#         for v in feature_scores.values():
#             scores.append(v)
#
#         # select the indices of the n_features best scores
#         idx = np.argpartition(scores, n_features)[-n_features:]
#         # pick from the dataset only the n_features features at the indeces idx
#         new_dataset = Data(n_datapoints=self.n_datapoints, n_features=n_features, features=list(np.zeros(n_features)))
#         new_dataset.data = np.zeros((new_dataset.n_datapoints, new_dataset.n_features))
#         for i in range(self.n_datapoints):
#             for k, j in enumerate(idx):
#                 new_dataset.data[i, k] = self.data[i, j]
#                 new_dataset.features[k] = self.features[j]
#
#         print(new_dataset.features)
#         return new_dataset
#
#     def normalization(self):
#         """
#         normalization
#
#         This function normalizes the dataset between -1,1 without
#         changing the mean and the standard deviation.
#
#         self: dataset to be normalized
#         """
#
#         transformer = MaxAbsScaler().fit(self.data)
#         self.data = transformer.transform(self.data)
#         return self
#
#     def dataset_truncation(self, N_TRAIN, label, datatype, label_type):
#         N_TEST = round(N_TRAIN * 0.2)
#         x = Data()
#         y = Data()
#         if datatype == 'train':
#             x.n_datapoints = N_TRAIN
#             x.n_features = self.n_features
#             y.n_datapoints = N_TRAIN
#             i = 0
#         if datatype == 'test':
#             x.n_datapoints = N_TEST
#             x.n_features = self.n_features
#             y.n_datapoints = N_TEST
#             i = N_TRAIN
#
#         x.data = []
#         y.data = []
#
#         k = 0
#         if datatype == 'train':
#             dim = round(N_TRAIN / 2)
#
#         elif datatype == 'test':
#             dim = round(N_TEST / 2)
#
#         while k < dim:
#
#             if label[i] == 0:
#                 if label_type == 'classical':
#                     x.data.append(self.data[i])
#                     y.data.append(0)
#                     k += 1
#                 elif label_type == 'quantum':
#                     x.data.append(self.data[i])
#                     y.data.append(-1)
#                     k += 1
#
#             i += 1
#
#         k = 0
#         if datatype == 'train':
#             i = 0
#         elif datatype == 'test':
#             i = N_TRAIN
#
#         while k < dim:
#
#             if label[i] == 1:
#                 x.data.append(self.data[i])
#                 y.data.append(1)
#                 k += 1
#             i += 1
#
#         x.data = np.array(x.data)
#         y.data = np.array(y.data)
#         return x, y
#
#     def data_save(self):
#         return np.savetxt(self.path + self.filename, self.data)
#
#
# def download_dataset_HEP(datatype, N_TRAIN, n_features):
#     X = Data()
#     y = Data()
#
#     X.filename = f'x_{datatype}_higgs_{N_TRAIN}_{n_features}'
#     X.path = 'resources/'
#     y.filename = f'y_{datatype}_higgs_{N_TRAIN}_{n_features}'
#     y.path = 'resources/'
#
#     X.data_load(datatype)
#     y.label_load(datatype)
#
#     print('uploaded')
#
#     X.get_feature_names()
#     y.get_feature_names()
#
#     print('===========================================')
#     print('\nDataset and labels loaded\n')
#     print('Initial shapes of the arrays are:')
#     print(f'dataset: {X.data.shape}')
#     print(f'labels: {y.data.shape}')
#     print('Computing roc curves for feature selection...\n')
#     feature_scores = X.compute_auc_roc(y.data)
#
#     X_feat_selected = X.feature_selection(feature_scores, n_features)
#     print(f'The dataset has now {n_features} features\n')
#     X_normalized = X_feat_selected.normalization()
#     print('Dataset normalized')
#     x, y = X_normalized.dataset_truncation(N_TRAIN, y.data, datatype, 'quantum')
#     print(f'The dataset has now {x.n_datapoints} datapoints\n')
#
#     X.data_save()
#     y.data_save()
#     return X, y


# def generate_HEP_dataset_quantum(datatype, N_TRAIN, n_features, enc):
#     X, y = download_dataset_HEP(datatype, N_TRAIN, n_features)
#     y = observables(datatype, N_TRAIN, n_features, enc, pennylane=False, save=True)
#     return X, y


# use in future version to generate dataset quantum from classical datasets
# def generate_dataset_quantum(the_id, wires):
#     X, y = download_dataset_openml(the_id)
#     y = random_qnn_encoding(X, wires)
#     return X, y
