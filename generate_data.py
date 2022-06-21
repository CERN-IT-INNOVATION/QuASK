import click
from quask.datasets import generate_dataset_quantum, create_HEP_dataset

@click.group()
def main():
    """LEAVE THE MAIN ALOOOOONEEEEEEE"""
    pass

@main.command()
def interactive():
    """
    Interactive prompt guiding the user through the generation of the dataset
    :return: None
    """
    # Which dataset do you want to generate among the proposed oness?
    # press 1 for generate_dataset_quantum
    # press 2 for create_dataset
    pass


@main.command()
def generate_dataset(index):
    """
    Interactive prompt guiding the user through the generation of the dataset
    :return: None
    """
    # Which dataset do you want to generate among the proposed oness?
    # press 1 for generate_dataset_quantum
    # press 2 for create_dataset_HEP
    pass

if __name__ == '__main__':
    main()
