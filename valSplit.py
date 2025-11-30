# src/create_val_split.py
import pandas as pd

def create_val_split():
    # 1. Load original full data
    df = pd.read_csv('data/nyu2_train.csv', header=None)
    
    # 2. Reproduce the training split logic to find which indices were used
    # (We must use the SAME random_state=42 as before)
    train_subset = df.sample(n=30000, random_state=42)
    
    # 3. Drop those rows to get the "Remaining" data
    remaining_df = df.drop(train_subset.index)
    
    # 4. Sample 1000 images for validation (Speed up evaluation)
    val_subset = remaining_df.sample(n=1000, random_state=42)
    
    # 5. Save
    val_subset.to_csv('data/val_split.csv', index=False, header=False)
    print(f"Created data/val_split.csv with {len(val_subset)} images.")

if __name__ == "__main__":
    create_val_split()