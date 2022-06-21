import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from scipy import stats
import click
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import prince
from .datasets import the_dataset_register
from .metrics import calculate_generalization_accuracy
from .kernels import the_kernel_register


@click.group()
def main():
    """The QuASK script ti aiuta a farti le seghe"""
    pass


@main.command()
def get_dataset():
    """
    Guides the user through the generation of some dataset
    """
    print("The available datasets are:")
    for i, dataset_specs in enumerate(the_dataset_register):
        print(f"{i:3d}: {dataset_specs['name']:30s} - {dataset_specs['type']:15s} - {dataset_specs['information']:30s}")
    index = click.prompt("Which dataset will you generate?", type=click.IntRange(0, len(the_dataset_register) - 1))
    path = click.prompt("Where is the output folder?", type=click.Path(exists=True, file_okay=False, dir_okay=True))
    dataset_specs = the_dataset_register.datasets[index]
    X, y = dataset_specs['get_dataset']()
    name = dataset_specs['name']
    np.save(f"{path}/{name}_X.npy", X)
    np.save(f"{path}/{name}_y.npy", y)
    print(f"Saved file {path}/{name}_X.npy")
    print(f"Saved file {path}/{name}_y.npy")


@main.command()
def preprocess_dataset():
    # reproducibility seed
    seed = click.prompt('Choose a random seed', type=int)
    np.random.seed(seed)
    # split training and testing set percentage
    train_test_split_seed = click.prompt('Choose a random seed for splitting training and testing set', type=int)
    # choose output location
    save_path = click.prompt("Where is the output folder", type=click.Path(exists=True, file_okay=False, dir_okay=True))
    # load feature data
    X_path = click.prompt('Specify the path of classical feature data X (.npy array)', type=click.Path(exists=True))
    X = np.load(X_path)
    print(f"\tLoaded X having shape {X.shape} ({X.shape[0]} samples of {X.shape[1]} features)")
    # load labels
    Y_path = click.prompt('Specify the path of classical feature data T (.npy array)', type=click.Path(exists=True))
    Y = np.load(Y_path)
    print(f"\tLoaded Y having shape {Y.shape} ({Y.shape[0]} labels)")
    # distinguish between classification and regression tasks
    is_classification = Y.dtype != 'float'
    print(f"\tLabels have type {Y.dtype} thus this is a {'classification' if is_classification else 'regression'} problem")
    # preprocess classification task
    if is_classification:
        # get classes
        classes = list(set(Y))
        classes.sort()
        assert len(classes) >= 2, "You can't have dataset with less than 2 classes"
        # print information about the class distribution
        print(f"\tThe dataset has {len(classes)} labels being {classes} distributed as it follows:")
        print(Counter(Y))
        for item, counts in Counter(Y).items():
            print(f"\t\tClass {item} is present {counts} times ({100 * counts / len(Y):4.2f} %)")
        # allow to pick only come classes while discarding others
        if len(classes) > 2 and click.confirm('Do you want to pick just the first two classes?'):
            indexes = Y[(Y == classes[0]) | (Y == classes[1])]
            X = X[indexes]
            Y = Y[indexes]
        # undersampling
        if click.confirm(f'Do you want to undersample the largest class?'):
            pass  # TODO
            # perc = click.prompt(f'The smaller class is x% the size of the larger class, x = ', type=click.FloatRange(0.0, 1.0))
            # X, Y = RandomOverSampler(sampling_strategy=perc).fit_resample(X, Y)
        # oversampling
        if click.confirm(f'Do you want to oversample the smallest class?'):
            pass  # TODO
            # perc = click.prompt(f'The smaller class is x% the size of the larger class, x = ', type=click.FloatRange(0.0, 1.0))
            # X, Y = RandomUnderSampler(sampling_strategy=perc).fit_resample(X, Y)
    else:
        # show statistics about the (real valued) labels
        desc = stats.describe(Y)
        print(f"\tThe labels ranges in {desc.minmax} with mean {desc.mean} and variance {desc.variance}")
    # feature preprocessing
    if click.confirm(f'Do you want to apply preprocessing to the features?'):
        if click.confirm(f'Do you want to apply PCA (numerical data only)?'):
            n_components = click.prompt("How many components you want?", type=click.IntRange(1, X.shape[1]))
            X = PCA(n_components=n_components).fit_transform(X)
        if click.confirm(f'Do you want to apply FAMD (both numerical and categorical data)?'):
            n_components = click.prompt("How many components you want?", type=click.IntRange(1, X.shape[1]))
            X = prince.FAMD(n_components=n_components, n_iter=10, copy=True).fit_transform(X)
        if click.confirm(f'Do you want to scale each field from 0 to 1 (MinMaxScaler)?'):
            X = MinMaxScaler().fit_transform(X)
    # training and testing dataset split
    test_perc = click.prompt(f'Which percentage of data must be in the test set?', type=click.FloatRange(0.0, 1.0))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_perc, random_state=train_test_split_seed)
    # save dataset files
    np.save(f"{save_path}/X_train.npy", X_train)
    np.save(f"{save_path}/X_test.npy", X_test)
    np.save(f"{save_path}/Y_train.npy", Y_train)
    np.save(f"{save_path}/Y_test.npy", Y_test)
    print(f"Saved file {save_path}/X_train.npy")
    print(f"Saved file {save_path}/y_train.npy")
    print(f"Saved file {save_path}/X_test.npy")
    print(f"Saved file {save_path}/y_test.npy")


@main.command()
def apply_kernel():
    # load files
    X_train_path = click.prompt("Where is X_train (npy file)?", type=click.Path(exists=True, file_okay=True, dir_okay=False))
    y_train_path = click.prompt("Where is y_train (npy file)?", type=click.Path(exists=True, file_okay=True, dir_okay=False))
    X_test_path = click.prompt("Where is X_test (npy file)?", type=click.Path(exists=True, file_okay=True, dir_okay=False))
    y_test_path = click.prompt("Where is y_test (npy file)?", type=click.Path(exists=True, file_okay=True, dir_okay=False))
    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    # fixed or trainable kernels?
    if click.confirm('Do you want to apply a fixed kernel [Y] or a trainable one [N]?'):
        print("The available fixed kernels are:")
        # list the kernels
        for i, (kernel_fn, kernel_name, kernel_params_list) in enumerate(the_kernel_register):
            print(f"{i:3d}: {kernel_name:40s}")
        # choose a single kernel from the list
        index = click.prompt("Which kernel Gram matrix will you generate?", type=click.IntRange(0, len(the_dataset_register) - 1))
        kernel_name = the_kernel_register.kernel_names[index]
        kernel_fn = the_kernel_register.kernel_functions[index]
        kernel_params_list = the_kernel_register.parameters[index]
        params = []
        # get optional hyper-parameters
        for param in kernel_params_list:
            the_param_value = click.prompt(f"Insert hyper-parameter {param}", type=click.FloatRange(0.0, 1000.0))
            params.append(the_param_value)
        # calculate and save
        path = click.prompt("Where is the output folder?", type=click.Path(exists=True, file_okay=False, dir_okay=True))
        training_gram = kernel_fn(X_train, X_train, params)
        testing_gram = kernel_fn(X_train, X_test, params)
        np.save(f"{path}/training_{kernel_name}.npy", training_gram)
        np.save(f"{path}/testing_{kernel_name}.npy", testing_gram)
        print(f"Saved file {path}/training_{kernel_name}.npy")
        print(f"Saved file {path}/testing_{kernel_name}.npy")
    else:
        pass



@main.command()
@click.option('--x', type=click.Path(exists=True), required=True, multiple=True)
@click.option('--y', type=click.Path(exists=True), required=True, multiple=True)
@click.option('--xt', type=click.Path(exists=True), required=True, multiple=True)
@click.option('--yt', type=click.Path(exists=True), required=True, multiple=True)
@click.option('--label', type=click.STRING, required=True, multiple=True)
def plot_accuracy(x, y, xt, yt, label):
    """
    Generate the bar plot with the given input files.
    TODO TEST.
    TODO ADD ERROR BAR FOR ELEMENTS WITH THE SAME LABEL.
    :param x: Training gram matrix file (.npy, can be repeated multiple times)
    :param y: Training label file (.npy, can be repeated multiple times)
    :param xt: Testing gram matrix file (.npy, can be repeated multiple times)
    :param yt: Testing label file (.npy, can be repeated multiple times)
    :param label: Label of the current bar (string)
    :return: None, shows a matplotlib image on a new window
    """
    ind = range(len(x))
    acc = [calculate_generalization_accuracy(xc, yc, xtc, ytc) for (xc, yc, xtc, ytc) in zip(x, y, xt, yt)]
    fig, ax = plt.subplots()
    p1 = ax.bar(ind, acc, 0.35, yerr=0, label='')
    ax.set_xlabel('Models')
    ax.set_xticks(ind, labels=label)
    ax.set_ylabel('Accuracy')
    ax.set_ylim((0, 1))
    ax.set_title('Accuracies of the configurations')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
