import openml


def download_dataset_openml(the_id):
    """
    Download a dataset from OpenML platform given the ID of the dataset
    :param the_id: ID of the dataset
    :return: X, y tuple
    """
    metadata = openml.datasets.get_dataset(the_id)
    # get dataset
    X, y, _, attribute_names = metadata.get_data(dataset_format="array", target=metadata.default_target_attribute)
    return X, y


def download_dataset_openml_by_name(name):
    """
    Download a dataset from OpenML platform given the name of the dataset
    :param name: name of the dataset
    :return: X, y tuple
    """
    # get the list of all datasets in OpenML platform
    openml_df = openml.datasets.list_datasets(output_format="dataframe")
    # retrieve the dataset id
    the_id = int(openml_df.query(f'name == "{name}"').iloc[0]["did"])
    return download_dataset_openml_by_name(the_id)


class DatasetRegister:

    def __init__(self):
        self.datasets = []
        self.current = 0

    def register(self, dataset_name, dataset_type, information_nature, get_dataset):
        """
        Register a new dataset
        :param dataset_name: name of the dataset
        :param dataset_type: 'regression' or 'classification'
        :param information_nature: 'classical' or 'quantum'
        :param get_dataset: function pointer to a zero-parameter function returning (X, y)
        :return: None
        """
        assert dataset_type in ['regression', 'classification']
        assert information_nature in ['classical', 'quantum']
        self.datasets.append({
            'name': dataset_name,
            'type': dataset_type,
            'information': information_nature,
            'get_dataset': get_dataset
        })

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
the_dataset_register.register('iris', 'classification', 'classical', lambda: download_dataset_openml(61))
the_dataset_register.register('Fashion-MNIST', 'classification', 'classical', lambda: download_dataset_openml(40996))
the_dataset_register.register('liver-disorders', 'regression', 'classical', lambda: download_dataset_openml(8))
the_dataset_register.register('delta_elevators', 'regression', 'classical', lambda: download_dataset_openml(198))


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
#     def data_load(self):
#         """
#         load background/signal
#
#         This function loads the background and signal dataset.
#
#         """
#         # fetching the background dataset from its path
#
#         self.data = np.load(f'{self.path + self.filename}')
#         self.n_datapoints = self.data.shape[0]
#         self.n_features = self.data.shape[1]
#         return self
#
#     def label_load(self):
#         """
#         load background/signal
#
#         This function loads the background and signal dataset.
#
#         """
#         # fetching the background dataset from its path
#
#         self.data = np.load(f'{self.path + self.filename}')
#         self.n_datapoints = self.data.shape[0]
#         return self
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
# # ============================================
# # -----------------end-class------------------
# # ============================================
#
# def create_dataset(filename, path, filename_label, datatype, N_TRAIN, n_features, features):
#     X = Data(filename=filename, path=path, features=features)
#     X.data_load()
#     y = Data(filename=filename_label, path=path, features=features)
#     y.label_load()
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
#     x.filename = f'x_{datatype}_higgs_{N_TRAIN}_{n_features}'
#     x.path = '/data/fdimarca/projected_kernel/Higgs_data/'
#     y.filename = f'y_{datatype}_higgs_{N_TRAIN}_{n_features}'
#     y.path = '/data/fdimarca/projected_kernel/Higgs_data/'
#
#     x.data_save()
#     y.data_save()
#     return x, y
#
#
# def generate_dataset_quantum(filename, path, filename_label, datatype, N_TRAIN, n_features, enc, features):
#     X, y = create_dataset(filename, path, filename_label, datatype, N_TRAIN, n_features=n_features, features=features)
#     y = observables(datatype, N_TRAIN, n_features, enc, pennylane=False, save=True)
#     return X, y
#
#