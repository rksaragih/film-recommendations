import pandas as pd
import ast

DATASET_PATH = "../Dataset/"

credits = pd.read_csv(DATASET_PATH + "credits.csv", low_memory=False)

# Cek apakah id 862 (Toy Story) ada di credits
print(credits[credits['id'] == 862])