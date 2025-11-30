import pandas as pd
import numpy as np
import os

def create_curriculum_splits(source_csv, output_dir, total_samples=30000, chunk_size=5000):
    # 1. Load the full dataset
    df = pd.read_csv(source_csv, header=None)
    print(f"Original Dataset Size: {len(df)}")

    # 2. Randomly sample 30k images
    # We use a fixed random_state for reproducibility
    df_subset = df.sample(n=total_samples, random_state=42).reset_index(drop=True)
    print(f"Subset Size: {len(df_subset)}")

    # 3. Split into chunks of 5k
    # 30,000 / 5,000 = 6 chunks
    num_chunks = total_samples // chunk_size
    chunks = np.array_split(df_subset, num_chunks)

    # 4. Save each chunk
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, chunk in enumerate(chunks):
        save_path = os.path.join(output_dir, f'train_chunk_{i}.csv')
        chunk.to_csv(save_path, index=False, header=False)
        print(f"Saved {save_path} with {len(chunk)} images.")

if __name__ == "__main__":
    # Adjust paths as needed
    create_curriculum_splits(
        source_csv='nyu/nyu_data/data/nyu2_train.csv', 
        output_dir='data/splits/'
    )