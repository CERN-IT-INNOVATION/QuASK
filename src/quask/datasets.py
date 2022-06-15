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


def generate_quantum_labels():
    X, y = None, None
    return X, y


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
