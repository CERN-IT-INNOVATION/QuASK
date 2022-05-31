import numpy as np
from collections import Counter
from scipy import stats
import click
from sklearn.model_selection import train_test_split
from quask.kernels import the_kernel_register


@click.group()
def main():
    """LEAVE THE MAIN ALOOOOONEEEEEEE"""
    pass


@main.command()
def analyze():
    """
    Interactive prompt guiding the user through the analysis of its dataset
    :return: None
    """
    # reproducibility seed
    seed = click.prompt('Choose a random seed:', type=int)
    np.random.seed(seed)

    # load feature data
    X_path = click.prompt('Specify the path of classical feature data X (.npy array):', type=click.Path(exists=True))
    X = np.load(X_path.name)
    print(f"\tLoaded X having shape {X.shape} ({X.shape[0]} samples of {X.shape[1]} features)")

    # load labels
    Y_path = click.prompt('Specify the path of classical feature data T (.npy array):', type=click.Path(exists=True))
    Y = np.load(Y_path.name)
    print(f"\tLoaded Y having shape {Y.shape} ({Y.shape[0]} labels)")

    # distinguish between classification and regression tasks
    is_classification = Y.dtype == 'float'
    print(f"\tLabels have type {Y.dtype} thus this is a {'classification' if is_classification else 'regression'} problem")

    # preprocess classification task
    if is_classification:
        # get classes
        classes = list(set(Y))
        classes.sort()

        # print information about the class distribution
        print(f"\tThe dataset has {len(classes)} labels being {classes} distributed as it follows:")
        for item, counts in Counter(classes):
            print(f"\t\tClass {item} is present {counts} times ({counts/len(classes)} %)")
        assert len(classes) >= 2, "You can't have dataset with less than 2 classes"

        # allow to pick only come classes while discarding others
        if len(classes) > 2 and click.confirm('Do you want to pick just some of these classes?'):
            pass  # TODO

        # undersampling
        for cclass in classes:
            if click.confirm(f'Do you want to undersample class {cclass}?'):
                perc = click.prompt(f'Which percentage of class {cclass} you want to keep?', type=click.FloatRange(0.0, 1.0))
                # TODO apply undersample

        # oversampling
        for cclass in classes:
            if click.confirm(f'Do you want to oversample class {cclass}?'):
                perc = click.prompt(f'Which percentage of class {cclass} you want to repeat?', type=click.FloatRange(1.0, 10.0))
                # TODO apply oversample
    else:
        # show statistics about the (real valued) labels
        desc = stats.describe(Y)
        print(f"\tThe labels ranges in {desc.minmax} with mean {desc.mean} and variance {desc.variance}")

    # feature preprocessing
    if click.confirm(f'Do you want to apply preprocessing to the features?'):
        if click.confirm(f'Do you want to apply PCA before scaling the fields?'):
            # TODO apply PCA
            pass
        if click.confirm(f'Do you want to scale each field from 0 to 1?'):
            # TODO apply MinMaxScaler
            pass
        if click.confirm(f'Do you want to apply PCA after scaling the fields?'):
            # TODO apply PCA
            pass

    # training and testing dataset split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

    # save dataset files
    save_path = click.prompt("Tell me where to save the processed dataset:", type=click.Path(exists=True, file_okay=False, dir_okay=True))
    np.save(f"{save_path}/X_train.npy", X_train)
    np.save(f"{save_path}/X_test.npy", X_test)
    np.save(f"{save_path}/Y_train.npy", Y_train)
    np.save(f"{save_path}/Y_test.npy", Y_test)

    # apply non-trainable kernels
    for (kernel_fn, kernel_name, kernel_params) in the_kernel_register:
        if click.confirm(f'Do you want to generate the Gram matrix for the {kernel_name}?'):
            params = []
            for i in range(kernel_params):
                param = click.prompt("Insert {i+1}-th param:", type=float)
                params.append(param)
            params_str = "_".join([str(p) for p in params])  # generate parameters identifier
            params_str = ''.join(e for e in params_str if e.isalnum())  # remove non alphanum characters
            gm_train = kernel_fn(X_train, None, params)
            gm_test = kernel_fn(X_test, X_train, params)
            np.save(f"{save_path}/{kernel_name}_{params_str}_gm_train.npy", gm_train)
            np.save(f"{save_path}/{kernel_name}_{params_str}_gm_test.npy", gm_test)

    # apply trainable kernels
    # TODO

