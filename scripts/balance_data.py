import pandas as pd
import numpy as np
import os
import sys

def balance_dataset(input_file, output_file, target_count=100):
    """
    Reads a CSV, balances classes to a specific count, and saves the result.
    """
    print(f"--- Processing {input_file} ---")
    
    # 1. Load Data
    if not os.path.exists(input_file):
        print(f"Error: The file '{input_file}' was not found in the current directory.")
        return

    try:
        # Using engine='python' to handle complex quoting if necessary, though default is usually fine
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Basic validation
    if 'label' not in df.columns or 'text' not in df.columns:
        print("Error: Input CSV must contain 'text' and 'label' columns.")
        print(f"Found columns: {list(df.columns)}")
        return

    # 2. Pre-processing
    initial_count = len(df)
    
    # Remove rows with empty text or labels
    df = df.dropna(subset=['text', 'label'])
    
    # Ensure text is string and strip whitespace
    df['text'] = df['text'].astype(str).str.strip()
    
    # Remove duplicates to ensure "distinctive" data
    df = df.drop_duplicates(subset=['text'])
    
    dedup_count = len(df)
    print(f"Original rows: {initial_count}")
    print(f"After removing duplicates/NAs: {dedup_count}")
    print("-" * 30)

    # 3. Balancing Logic
    balanced_dfs = []
    
    # Get unique labels
    labels = df['label'].unique()
    labels.sort() # Sort for consistent output
    
    print(f"Found {len(labels)} classes: {labels}")
    print("-" * 30)
    print(f"{'Label':<10} | {'Available':<10} | {'Action':<15}")
    print("-" * 30)

    for label in labels:
        class_subset = df[df['label'] == label]
        count = len(class_subset)
        
        if count >= target_count:
            # Downsample: Randomly select target_count
            # We use a fixed random_state for reproducibility
            action = "Downsampling"
            sampled = class_subset.sample(n=target_count, random_state=42)
        else:
            # Upsample: Sample with replacement to fill the gap
            # This ensures we hit exactly 50, even if we have to repeat some distinctive examples
            action = f"Upsampling (x{target_count/count:.1f})"
            sampled = class_subset.sample(n=target_count, replace=True, random_state=42)
            
        print(f"{str(label):<10} | {count:<10} | {action:<15}")
        balanced_dfs.append(sampled)

    # 4. Combine and Shuffle
    final_df = pd.concat(balanced_dfs)
    
    # Shuffle the final dataset so classes aren't clustered
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 5. Save
    try:
        final_df.to_csv(output_file, index=False)
        print("-" * 30)
        print(f"Success! Balanced dataset saved to: {output_file}")
        print(f"Total rows: {len(final_df)}")
        print(f"Examples per class: {target_count}")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    INPUT_FILENAME = 'data/balanced_dataset_large.csv'
    OUTPUT_FILENAME = 'data/training.csv'
    
    balance_dataset(INPUT_FILENAME, OUTPUT_FILENAME)