import click
import numpy as np
from quask.datasets import the_dataset_register


@click.group()
def main():
    """The script is used to retrieve or generate a new dataset"""
    pass


@main.command()
def interactive():
    """
    Interactive prompt guiding the user through the generation of the dataset
    :return: None
    """
    print("The available datasets are:")
    for i, dataset_specs in enumerate(the_dataset_register):
        print(f"{i:3d}: {dataset_specs['name']:30s} - {dataset_specs['type']:15s} - {dataset_specs['information']:30s}")
    index = click.prompt("Which dataset will you generate?", type=click.IntRange(0, len(the_dataset_register)-1))
    path = click.prompt("Where is the output folder?", type=click.Path(exists=True, file_okay=False, dir_okay=True))
    dataset_specs = the_dataset_register.datasets[index]
    X, y = dataset_specs['get_dataset']()
    name = dataset_specs['name']
    np.save(f"{path}/{name}_X.npy", X)
    np.save(f"{path}/{name}_y.npy", y)
    print("Completed!")


@main.command()
@click.option('--name', type=str, required=True)
@click.option('--output-folder', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True)
def generate_dataset(name, output_folder):
    dataset_specs_list = [ds for ds in the_dataset_register if ds['name'] == name]
    if len(dataset_specs_list) == 0:
        print("Dataset not found")
    dataset_specs = dataset_specs_list[0]
    X, y = dataset_specs['get_dataset']()
    name = dataset_specs['name']
    np.save(f"{output_folder}/{name}_X.npy", X)
    np.save(f"{output_folder}/{name}_y.npy", y)


if __name__ == '__main__':
    main()
