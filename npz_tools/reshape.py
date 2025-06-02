import pandas as pd  # Import pandas to read the CSV file
import numpy as np

def preprocess_dna_sequence(encoded_seq, target_length=224):
    """
    Preprocesses a DNA sequence into a tensor of size [4, 224, 224].
    Each sequence is one-hot encoded and then resized/padded to 224 length.
    """
    # Check if the length of encoded_seq is divisible by 4
    if len(encoded_seq) % 4 != 0:
        raise ValueError("The length of the encoded sequence must be divisible by 4")
    
    # Convert the flattened one-hot encoded string back into a binary array
    binary_matrix = np.array([int(b) for b in encoded_seq], dtype=np.int8).reshape(-1, 4)

    # Step 1: Pad or truncate to match 224 nucleotides
    current_length = binary_matrix.shape[0]
    if current_length < target_length:
        pad_size = target_length - current_length
        padded_matrix = np.pad(binary_matrix, ((0, pad_size), (0, 0)), mode='constant')
    else:
        padded_matrix = binary_matrix[:target_length, :]

    # Step 2: Reshape to [4, 224, 224] by repeating along the second dimension
    reshaped_array = np.stack([padded_matrix.T] * target_length, axis=-1)  # Shape [4, 224, 224]

    return reshaped_array

def process_entire_file_with_integer_labels(input_file, output_file):
    """
    Processes a CSV file with DNA sequences and their labels, 
    reshapes the sequences to [4, 224, 224] format, and saves to .npz.
    """
    # Load the CSV file
    df = pd.read_csv(input_file)
    
    # Create a mapping for labels
    label_mapping = {'healthy': 0, 'WSSV': 1, 'AHPND': 2}
    
    # Prepare arrays for reshaped data and labels
    reshaped_data = []
    labels = []
    
    for idx, row in df.iterrows():
        encoded_seq = row['Encoded_DNA']
        status = row['status']
        
        # Reshape each sequence into [4, 224, 224]
        reshaped_seq = preprocess_dna_sequence(encoded_seq)
        reshaped_data.append(reshaped_seq)
        
        # Store the label as an integer
        labels.append(label_mapping[status])
    
    # Convert to numpy arrays for storage
    reshaped_data = np.array(reshaped_data, dtype=np.int8)
    labels = np.array(labels, dtype=np.int8)
    
    # Save reshaped data and labels in a .npz file
    np.savez(output_file, dna_sequences=reshaped_data, labels=labels)
    
    print(f"Reshaped data and labels saved to {output_file}")

# Example usage
input_file = '/home/user/torch_shrimp/dataset/Mixed/Test/5101/5101_onehot/Test5101_encode.csv'
output_file = '/home/user/torch_shrimp/until-tools/mod/test5101.npz'
process_entire_file_with_integer_labels(input_file, output_file)
