from quask.metrics import calculate_generalization_accuracy
import click


@click.group()
def main():
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
