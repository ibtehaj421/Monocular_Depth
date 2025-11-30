import pandas as pd
import numpy as np
import os

def create_extra_chunks():
    print("--- Identifying Unseen Data ---")
    # 1. Load Full Data
    df = pd.read_csv('data/nyu2_train.csv', header=None)
    print(f"Total Images: {len(df)}")

    # 2. Re-create the original 30k training set (Indices only)
    # We use random_state=42 to match your previous logic EXACTLY
    train_subset = df.sample(n=30000, random_state=42)
    
    # 3. Re-create the validation set
    remaining_df = df.drop(train_subset.index)
    val_subset = remaining_df.sample(n=1000, random_state=42)
    
    # 4. Extract the UNUSED data
    # (Total - Train30k - Val1k)
    unused_df = remaining_df.drop(val_subset.index)
    print(f"Already Used: {len(train_subset)} (Train) + {len(val_subset)} (Val)")
    print(f"New Unseen Images: {len(unused_df)}")

    # 5. Split into new chunks (Chunks 6, 7, 8, 9...)
    # We want chunks of ~5000 images to match your curriculum size
    chunk_size = 5000
    num_new_chunks = int(np.ceil(len(unused_df) / chunk_size))
    
    chunks = np.array_split(unused_df, num_new_chunks)
    
    output_dir = 'data/splits/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Start numbering from 6 (since you already have 0-5)
    start_chunk_idx = 6
    
    for i, chunk in enumerate(chunks):
        chunk_idx = start_chunk_idx + i
        save_path = os.path.join(output_dir, f'train_chunk_{chunk_idx}.csv')
        chunk.to_csv(save_path, index=False, header=False)
        print(f"-> Created {save_path} with {len(chunk)} images.")

if __name__ == "__main__":
    create_extra_chunks()