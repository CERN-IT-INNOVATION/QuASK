import click
import numpy as np
from .datasets import the_dataset_register


@click.group()
def main():
    """The QuASK script ti aiuta a farti le seghe"""
    pass


@main.command()
def generate_dataset():
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
def process():
    pass


if __name__ == '__main__':
    main()
