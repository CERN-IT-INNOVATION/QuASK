import matplotlib.pyplot as plt
from quask.metrics import calculate_generalization_accuracy
import click


@click.group()
def main():
    """LEAVE THE MAIN ALOOOOONEEEEEEE"""
    pass


@main.command()
@click.option('--training-gram', type=click.Path(exists=True), required=True)
@click.option('--training-labels', type=click.Path(exists=True), required=True)
@click.option('--testing-gram', type=click.Path(exists=True), required=True)
@click.option('--testing-labels', type=click.Path(exists=True), required=True)
def calculate_accuracy(training_gram, training_labels, testing_gram, testing_labels):
    """
    Calculate the accuracy of a previously generated Gram matrix for training and testing set
    :param training_gram: Gram matrix of the training set, must have shape (N,N)
    :param training_labels: Labels of the training set, must have shape (N,)
    :param testing_gram: Gram matrix of the testing set, must have shape (M,N)
    :param testing_labels:Labels of the training set, must have shape (M,)
    :return: None
    """
    accuracy = calculate_generalization_accuracy(training_gram, training_labels, testing_gram, testing_labels)
    print(accuracy)


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
