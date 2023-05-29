"""This module allows to run quask as a command-line interface tool.

Example:
    To retrieve the datasets available:

        $ python3.9 -m quask get-dataset

    To preprocess a dataset:

        $ python3.9 -m quask preprocess-dataset

    To analyze a dataset using quantum and classical kernels:

        $ python3.9 -m apply-kernel

    To create some plot of the property related to the generated Gram matrices:

        $ python3.9 -m quask plot-metric --metric accuracy --train-gram training_linear_kernel.npy --train-y Y_train.npy --test-gram testing_linear_kernel.npy --test-y Y_test.npy --label linear

"""


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
from .metrics import (
    calculate_generalization_accuracy,
    calculate_kernel_target_alignment,
    calculate_geometric_difference,
    calculate_model_complexity,
)
from .kernels import the_kernel_register
from .template_pennylane import PennylaneTrainableKernel


@click.group()
def main():
    """Starts quask command-line interface tool."""
    pass


@main.command()
def get_dataset():
    """
    The script is used to retrieve one of the datasets available in quask.datasets
    module. The output of the process is a couple of NumPy binary files representing
    the feature data and the corresponding labels.

    Returns:
        None
    """
    print("The available datasets are:")
    for i, dataset_specs in enumerate(the_dataset_register):
        print(
            f"{i:3d}: {dataset_specs['name']:30s} - {dataset_specs['type']:15s} - {dataset_specs['information']:30s}"
        )
    index = click.prompt(
        "Which dataset will you generate?",
        type=click.IntRange(0, len(the_dataset_register) - 1),
    )
    path = click.prompt(
        "Where is the output folder?",
        type=click.Path(exists=True, file_okay=False, dir_okay=True),
    )
    dataset_specs = the_dataset_register.datasets[index]
    X, y = dataset_specs["get_dataset"]()
    name = dataset_specs["name"]
    np.save(f"{path}/{name}_X.npy", X)
    np.save(f"{path}/{name}_y.npy", y)
    print(f"Saved file {path}/{name}_X.npy")
    print(f"Saved file {path}/{name}_y.npy")


@main.command()
def preprocess_dataset():
    """
    The script can modify the dataset in several ways. Firstly, the user can vertically
    slice the dataset, keeping only a certain range of labels. Secondly, the user can
    apply dimensionality reduction techniques. These are especially important in
    the NISQ setting due to the lack of resources. The techniques available are
    PCA for numerical data and FAMD for mixed numerical and categorical data3.
    Thirdly, it is possible to fix the possible imbalanceness of the classes using
    random undersampling or random oversampling. When loading, the script
    already shows some statistics about the dataset, both for classification and
    regression tasks, which can guide the user through the preprocessing. The
    output of the process are the four files X train, y train, X test, y test which
    can now be fed to some kernel machine.

    Returns:
        None
    """
    # reproducibility seed
    seed = click.prompt("Choose a random seed", type=int)
    np.random.seed(seed)
    # split training and testing set percentage
    train_test_split_seed = click.prompt(
        "Choose a random seed for splitting training and testing set", type=int
    )
    # choose output location
    save_path = click.prompt(
        "Where is the output folder",
        type=click.Path(exists=True, file_okay=False, dir_okay=True),
    )
    # load feature data
    X_path = click.prompt(
        "Specify the path of classical feature data X (.npy array)",
        type=click.Path(exists=True),
    )
    X = np.load(X_path)
    print(
        f"\tLoaded X having shape {X.shape} ({X.shape[0]} samples of {X.shape[1]} features)"
    )
    # load labels
    Y_path = click.prompt(
        "Specify the path of classical feature data T (.npy array)",
        type=click.Path(exists=True),
    )
    Y = np.load(Y_path)
    print(f"\tLoaded Y having shape {Y.shape} ({Y.shape[0]} labels)")
    # distinguish between classification and regression tasks
    is_classification = Y.dtype != "float"
    print(
        f"\tLabels have type {Y.dtype} thus this is a {'classification' if is_classification else 'regression'} problem"
    )
    # preprocess classification task
    if is_classification:
        # get classes
        classes = list(set(Y))
        classes.sort()
        assert len(classes) >= 2, "You can't have dataset with less than 2 classes"
        # print information about the class distribution
        print(
            f"\tThe dataset has {len(classes)} labels being {classes} distributed as it follows:"
        )
        print(Counter(Y))
        for item, counts in Counter(Y).items():
            print(
                f"\t\tClass {item} is present {counts} times ({100 * counts / len(Y):4.2f} %)"
            )
        # allow to pick only come classes while discarding others
        if len(classes) > 2 and click.confirm(
            "Do you want to pick just the first two classes?"
        ):
            indexes = (Y == classes[0]) | (Y == classes[1])
            X = X[indexes]
            Y = Y[indexes]
        # undersampling
        if click.confirm(f"Do you want to undersample the largest class?"):
            pass  # TODO
            # perc = click.prompt(f'The smaller class is x% the size of the larger class, x = ', type=click.FloatRange(0.0, 1.0))
            # X, Y = RandomOverSampler(sampling_strategy=perc).fit_resample(X, Y)
        # oversampling
        if click.confirm(f"Do you want to oversample the smallest class?"):
            pass  # TODO
            # perc = click.prompt(f'The smaller class is x% the size of the larger class, x = ', type=click.FloatRange(0.0, 1.0))
            # X, Y = RandomUnderSampler(sampling_strategy=perc).fit_resample(X, Y)
    else:
        # show statistics about the (real valued) labels
        desc = stats.describe(Y)
        print(
            f"\tThe labels ranges in {desc.minmax} with mean {desc.mean} and variance {desc.variance}"
        )
    # feature preprocessing
    if click.confirm(f"Do you want to apply preprocessing to the features?"):
        if click.confirm(f"Do you want to apply PCA (numerical data only)?"):
            n_components = click.prompt(
                "How many components you want?", type=click.IntRange(1, X.shape[1])
            )
            X = PCA(n_components=n_components).fit_transform(X)
        if click.confirm(
            f"Do you want to apply FAMD (both numerical and categorical data)?"
        ):
            n_components = click.prompt(
                "How many components you want?", type=click.IntRange(1, X.shape[1])
            )
            X = prince.FAMD(
                n_components=n_components, n_iter=10, copy=True
            ).fit_transform(X)
        if click.confirm(
            f"Do you want to scale each field from 0 to 1 (MinMaxScaler)?"
        ):
            X = MinMaxScaler().fit_transform(X)
    # training and testing dataset split
    test_perc = click.prompt(
        f"Which percentage of data must be in the test set?",
        type=click.FloatRange(0.0, 1.0),
    )
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_perc, random_state=train_test_split_seed
    )
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
@click.option(
    "--x-train",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    required=False,
)
@click.option(
    "--y-train",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    required=False,
)
@click.option(
    "--x-test",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    required=False,
)
@click.option(
    "--y-test",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    required=False,
)
def apply_kernel(x_train, y_train, x_test, y_test):
    """
    The script is used to generate the gram matrices of the training and testing
    sets with respect to the selected kernel. The user can select to apply one of the
    kernels (classical and quantum) provided by the object the kernel register.
    The user can also select to apply a trainable quantum kernel. In this case,
    the user specifies the embedding and variational form (and its number of
    repetitions), the optimizer (ADAM stochastic gradient descent or grid search),
    and the reward function to maximize.

    Args:
        x_train: path of the training set feature data (the data must be in npy format)
        y_train: path of the testing set label data (the data must be in npy format)
        x_test: path of the testing set feature data (the data must be in npy format)
        y_test: path of the testing set label data (the data must be in npy format)

    Returns:
        None
    """
    # load files
    X_train_path = (
        click.prompt(
            "Where is X_train (npy file)?",
            type=click.Path(exists=True, file_okay=True, dir_okay=False),
        )
        if x_train is None
        else x_train
    )
    y_train_path = (
        click.prompt(
            "Where is y_train (npy file)?",
            type=click.Path(exists=True, file_okay=True, dir_okay=False),
        )
        if y_train is None
        else y_train
    )
    X_test_path = (
        click.prompt(
            "Where is X_test (npy file)?",
            type=click.Path(exists=True, file_okay=True, dir_okay=False),
        )
        if x_test is None
        else x_test
    )
    y_test_path = (
        click.prompt(
            "Where is y_test (npy file)?",
            type=click.Path(exists=True, file_okay=True, dir_okay=False),
        )
        if y_test is None
        else y_test
    )
    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    # fixed or trainable kernels?
    if click.confirm("Do you want to apply a fixed kernel [Y] or a trainable one [N]?"):
        print("The available fixed kernels are:")
        # list the kernels
        for i, (kernel_fn, kernel_name, kernel_params_list) in enumerate(
            the_kernel_register
        ):
            print(f"{i:3d}: {kernel_name:40s}")
        # choose a single kernel from the list
        index = click.prompt(
            "Which kernel Gram matrix will you generate?",
            type=click.IntRange(0, len(the_dataset_register) - 1),
        )
        kernel_name = the_kernel_register.kernel_names[index]
        kernel_fn = the_kernel_register.kernel_functions[index]
        kernel_params_list = the_kernel_register.parameters[index]
        params = []
        # get optional hyper-parameters
        for param in kernel_params_list:
            the_param_value = click.prompt(
                f"Insert hyper-parameter {param}", type=click.FloatRange(0.0, 1000.0)
            )
            params.append(the_param_value)
        # calculate and save
        path = click.prompt(
            "Where is the output folder?",
            type=click.Path(exists=True, file_okay=False, dir_okay=True),
        )
        print(X_train.shape, X_test.shape)
        training_gram = kernel_fn(X_train, X_train, params)
        testing_gram = kernel_fn(X_test, X_train, params)
        print(training_gram.shape, testing_gram.shape)
        np.save(f"{path}/training_{kernel_name}.npy", training_gram)
        np.save(f"{path}/testing_{kernel_name}.npy", testing_gram)
        print(f"Saved file {path}/training_{kernel_name}.npy")
        print(f"Saved file {path}/testing_{kernel_name}.npy")
    else:
        embedding = click.prompt(
            "Choose an embedding for your data",
            type=click.Choice(["rx", "ry", "rz", "zz"]),
        )
        var_form = click.prompt(
            "Choose an embedding for your data",
            type=click.Choice(["hardware_efficient", "tfim", "ltfim", "zz_rx"]),
        )
        layers = click.prompt("Choose a number of layers", type=click.IntRange(1, 1000))
        optimizer = click.prompt(
            "Choose an optimizer", type=click.Choice(["adam", "grid"])
        )
        metric = click.prompt(
            "Choose a reward metric to maximize",
            type=click.Choice(
                [
                    "kernel-target-alignment",
                    "accuracy",
                    "geometric-difference",
                    "model-complexity",
                ]
            ),
        )
        seed = click.prompt("Choose a random seed", type=int)

        np.random.seed(seed)
        trainable_kernel = PennylaneTrainableKernel(
            X_train,
            y_train,
            X_test,
            y_test,
            embedding,
            var_form,
            layers,
            optimizer,
            metric,
            seed=seed,
            keep_intermediate=True,
        )
        trainable_kernel.optimize_circuit()
        training_gram, testing_gram = trainable_kernel.get_optimized_gram_matrices()
        kernel_name = f"trainable_{embedding}_{var_form}_{layers}_{optimizer}_{metric}"
        path = click.prompt(
            "Where is the output folder?",
            type=click.Path(exists=True, file_okay=False, dir_okay=True),
        )
        np.save(f"{path}/training_{kernel_name}.npy", training_gram)
        np.save(f"{path}/testing_{kernel_name}.npy", testing_gram)
        print(f"Saved file {path}/training_{kernel_name}.npy")
        print(f"Saved file {path}/testing_{kernel_name}.npy")


@main.command()
@click.option(
    "--train-gram", type=click.Path(exists=True), required=True, multiple=True
)
@click.option("--train-y", type=click.Path(exists=True), required=True, multiple=True)
@click.option("--test-gram", type=click.Path(exists=True), required=True, multiple=True)
@click.option("--test-y", type=click.Path(exists=True), required=True, multiple=True)
@click.option("--label", type=click.STRING, required=True, multiple=True)
@click.option(
    "--metric",
    type=click.Choice(
        [
            "kernel-target-alignment",
            "accuracy",
            "geometric-difference",
            "model-complexity",
        ]
    ),
    required=True,
    multiple=False,
)
@click.option(
    "--geometric-difference-base-gram", type=click.Path(exists=True), required=False
)
def plot_metric(
    train_gram,
    train_y,
    test_gram,
    test_y,
    label,
    metric,
    geometric_difference_base_gram,
):
    """
    The script calculates the accuracy of a kernel machine using the training and
    testing Gram matrices given as input. The output is a plot comparing the
    different kernels. For each kernel matrix, the user specifies the label that appears
    at the x-axis of the plot. If multiple instances are specified with the same label
    these are interpreted as i.i.d. random experiments and will contribute to the
    error bars. The software also calculates, together with the accuracy, any metric
    present in quask.metrics.

    Args:
        train_gram: path of the training set Gram matrix (the data must be in npy format)
        train_y: path of the training set label data (the data must be in npy format)
        test_gram: path of the testing set Gram matrix (the data must be in npy format)
        test_y: path of the testing set label data (the data must be in npy format)
        label: name of the currently passed dataset
        metric: one of the following: kernel-target-alignment, accuracy, geometric-difference, model-complexity
        geometric_difference_base_gram: path of the training set Gram matrix used as a baseline for 'geometric-difference' metrics

    Returns:
        None. The output is an image save to relative path "plot_metric.png".
    """
    assert (
        len(train_gram) == len(train_y) == len(test_gram) == len(test_y) == len(label)
    )
    assert (
        metric != "geometric-difference" or geometric_difference_base_gram is not None
    )

    if geometric_difference_base_gram is not None:
        geometric_difference_base_gram = np.load(geometric_difference_base_gram)

    # get kernel names (do not sort?)
    kernel_names = list(dict.fromkeys(label))
    # create x axis
    ind = range(len(kernel_names))
    # create y axis
    kernel_values = []
    for i, kernel_name in enumerate(kernel_names):
        kernel_values.append([])
        for (
            train_gram_current,
            train_y_current,
            test_gram_current,
            test_y_current,
            kernel_name_current,
        ) in zip(train_gram, train_y, test_gram, test_y, label):
            train_gram_current = np.load(train_gram_current)
            train_y_current = np.load(train_y_current)
            test_gram_current = np.load(test_gram_current)
            test_y_current = np.load(test_y_current)
            if kernel_name_current == kernel_name:
                if metric == "kernel-target-alignment":
                    kernel_values[i].append(
                        calculate_kernel_target_alignment(
                            test_gram_current, test_y_current
                        )
                    )
                elif metric == "accuracy":
                    kernel_values[i].append(
                        calculate_generalization_accuracy(
                            train_gram_current,
                            train_y_current,
                            test_gram_current,
                            test_y_current,
                        )
                    )
                elif metric == "geometric-difference":
                    kernel_values[i].append(
                        calculate_geometric_difference(
                            test_gram_current, geometric_difference_base_gram
                        )
                    )
                elif metric == "model-complexity":
                    kernel_values[i].append(
                        calculate_model_complexity(train_gram_current, train_y_current)
                    )

    kernel_mean = [np.mean(item) for item in kernel_values]
    kernel_variance = [np.var(item) for item in kernel_values]
    # create plot
    fig, ax = plt.subplots()
    p1 = ax.bar(ind, kernel_mean, 0.35, yerr=kernel_variance, label="")
    ax.set_xlabel("Kernels")
    ax.set_xticks(ind, labels=kernel_names)
    ax.set_ylabel(metric)
    ax.set_ylim((0, 1))
    ax.set_title(f"{metric} of the configurations")
    plt.savefig("plot_metric.png")


if __name__ == "__main__":
    main()
