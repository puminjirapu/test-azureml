from dotenv import load_dotenv
import os

from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.core.exceptions import ResourceNotFoundError


def register_dataset(
    ml_client, 
    dataset_name: str, 
    dataset_path: str, 
    version: str = "1", 
    description: str = "", 
    tags: dict = None    
    ):
    """Registers a dataset in Azure ML if it doesn't already exist.

    Args:
        ml_client: The Azure MLClient instance.
        dataset_name (str): Name of the dataset.
        dataset_path (str): Path to the dataset in storage.
        version (str, optional): Version number of the dataset. Defaults to "1".
        description (str, optional): Description of the dataset. Defaults to "".
        tags (dict, optional): Metadata tags for the dataset. Defaults to {}.
    """
    if not dataset_name or not dataset_path:
        raise ValueError("Both 'dataset_name' and 'dataset_path' must be provided.")

    tags = tags or {}  # Default to empty dictionary if None
    try:
        ml_client.data.get(name=dataset_name, version=version)
        print(f"Dataset '{dataset_name}' (version {version}) already exists.")
    except ResourceNotFoundError:
        data_asset = Data(
            name=dataset_name,
            version=version,
            description=description,
            tags=tags,
            path=dataset_path,
            type=AssetTypes.URI_FILE,
        )
        ml_client.data.create_or_update(data_asset)
        print(f"Dataset '{dataset_name}' (version {version}) registered successfully.")
    