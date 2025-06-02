import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder

def is_dna(sequence):
    """Check if a given sequence is a valid DNA sequence."""
    valid_nucleotides = {'A', 'C', 'G', 'T'}
    return all(nucleotide in valid_nucleotides for nucleotide in sequence.upper())

def etl_dna_sequence_cleanup(dna_sequence_original_csv_path, dna_sequence_cleanup_csv_path, dna_sequence_columns_list):
    """Cleans up the original DNA sequence data and saves the cleaned data to a new CSV file.
    
    Args:
        dna_sequence_original_csv_path (string): Path to the original DNA sequence CSV file.
        dna_sequence_cleanup_csv_path (string): Path to save the cleaned DNA sequence CSV file.
        dna_sequence_columns_list (list): List of column names for the cleaned DataFrame (e.g., ["dna_class", "dna_sequence"]).
    
    Returns:
        bool: True if the cleanup process was successful, False otherwise.
    """ 
    result = False
    try:
        # Load the original DNA sequence data from CSV
        df_genomics_original = pd.read_csv(dna_sequence_original_csv_path)
        
        # Remove rows where all elements are missing
        df_genomics_original.dropna(how="all", inplace=True)
        print("DNA sequence original data frame shape:\n{} ".format(df_genomics_original.shape))
        
        # Initialize lists for the cleaned DNA data
        dna_class_list = []
        dna_sequence_list = []
        
        # Iterate over each row in the DataFrame
        for row in df_genomics_original.itertuples():
            dna_class = row.status
            dna_sequence = row.dna_sequence
            
            # Check if the sequence is a valid DNA sequence
            if is_dna(dna_sequence):
                dna_class_list.append(dna_class)
                dna_sequence_list.append(dna_sequence)
        
        # Create a new DataFrame with the cleaned data
        df_genomics_cleanup = pd.DataFrame(list(zip(dna_class_list, dna_sequence_list)), columns=dna_sequence_columns_list)
        print("DNA sequence cleanup data frame shape:\n{}".format(df_genomics_cleanup.shape))
        
        # Calculate and print the number of rows removed
        total_remove_rows = df_genomics_original.shape[0] - df_genomics_cleanup.shape[0]
        print("DNA sequence original total remove rows:\n{}".format(total_remove_rows))
        
        # Save the cleaned data to a new CSV file
        df_genomics_cleanup.to_csv(dna_sequence_cleanup_csv_path, index=False)
        print("DNA sequence cleanup CSV file created:\n{}".format(dna_sequence_cleanup_csv_path))
        
        result = True
    except Exception as e:
        print("An error occurred:", str(e))
    
    return result


# Example usage
original_path = "/home/user/torch_shrimp/dataset/healthy/"
clean_path = "/home/user/torch_shrimp/until-tools/"
dna_sequence_original_csv_path = os.path.join(original_path, "healthy_test.csv")
dna_sequence_cleanup_csv_path = os.path.join(clean_path, "clean_test.csv")
dna_sequence_columns_list = ["dna_class", "dna_sequence"]

result = etl_dna_sequence_cleanup(dna_sequence_original_csv_path, dna_sequence_cleanup_csv_path, dna_sequence_columns_list)
print(result)