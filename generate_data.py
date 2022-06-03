import click
from quask.datasets import generate_mio_dataset_quantum_fantastico

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
    # che dataset vuoi generare tra quelli proposti?
    # premi 1 per generate_mio_dataset_quantum_fantastico
    # premi 2 per ...
    pass


@main.command()
def generate_dataset(index):
    """
    Interactive prompt guiding the user through the generation of the dataset
    :return: None
    """
    # che dataset vuoi generare tra quelli proposti?
    # premi 1 per generate_mio_dataset_quantum_fantastico
    # premi 2 per ...
    pass
