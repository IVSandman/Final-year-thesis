import pandas as pd
import numpy as np

def one_hot_encode_dna(sequence):
    mapping = {'A': [1, 0, 0, 0],
               'T': [0, 1, 0, 0],
               'C': [0, 0, 1, 0],
               'G': [0, 0, 0, 1],
               'N' :[1,1,1,1]}
    try:
        return ''.join(str(x) for x in np.array([mapping[nucleotide] for nucleotide in sequence]).flatten())
    except KeyError as e:
        print(f"Unexpected nucleotide {e} found in the sequence: {sequence}")
        return None

file_path = '/home/user/torch_shrimp/dataset/WSSV/WSSV100/WSSV100.csv'  # Replace with your file path
output_path = '/home/user/torch_shrimp/dataset/WSSV/WSSV100/WSSV100_encode.csv'

try:
    df = pd.read_csv(file_path)

    # Apply one-hot encoding and store in a single column
    df['Encoded_DNA'] = df['dna_sequence'].apply(one_hot_encode_dna)

    # Verify the output
    print(df.head())

    # Save the encoded DataFrame to CSV
    df.to_csv(output_path, index=False)
    print(f"Encoded data saved successfully to {output_path}")
except FileNotFoundError:
    print("The file path provided is incorrect or the file does not exist.")
except Exception as e:
    print(f"An error occurred: {e}")
