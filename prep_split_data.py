import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data

from dotenv import load_dotenv

load_dotenv()

credential = DefaultAzureCredential()
try:
    ml_client = MLClient(
        credential=credential,
        subscription_id=os.getenv('SUBSCRIPTION_ID'),
        resource_group_name=os.getenv('RESOURCE_GROUP'),
        workspace_name=os.getenv('WORKSPACE_NAME')
    )

except Exception as ex:
    print("HERE IN THE EXCEPTION BLOCK")
    print(ex)

data_asset_name = "TEST_DATASET"
data_asset = ml_client.data.get(name=data_asset_name, version="1")

df = pd.read_csv(data_asset.path)
df_prep = df.drop(columns=['Posted On', 'Area Locality'])
df_prep['Floor'] = df_prep['Floor'].str.replace('Ground', '1')
df_prep['Floor_num'] = df_prep['Floor'].apply(lambda x: x.split(' out of ')[0])
df_prep['Floor_building'] = df_prep['Floor'].apply(lambda x: int(x.split(' out of ')[-1]))
df_prep.loc[df_prep['Floor_num'].str.contains('Basement'), 'Floor_num'] = '0'
df_prep['Floor_num'] = df_prep['Floor_num'].astype(int)
df_prep = df_prep.drop(columns='Floor')

train, test = train_test_split(df_prep, test_size=0.2, random_state=42)
test = test.drop(columns='Rent')

save_path = "testset.csv"
test.to_csv(save_path, index=False)

test_data_asset = Data(
    name='TEST_DATASET_test',
    version='1',
    description='Test set of TEST_DATASET data',
    type='uri_file',
    path=save_path,
)

ml_client.data.create_or_update(test_data_asset)
print(f"Data asset '{data_asset.name}' registered successfully!")