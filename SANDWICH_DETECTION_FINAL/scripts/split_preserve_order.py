import pandas as pd
import numpy as np

# Read original perfect dataset
df = pd.read_csv('Sandwich_and_no_dataset.csv')
print(f'Total rows in original: {len(df):,}')

# Verify it has proper ordering
print('\nChecking for consecutive sandwich triplets...')
found_count = 0
for i in range(min(1000, len(df)-2)):
    if df.iloc[i]['label']==1 and df.iloc[i+1]['label']==2 and df.iloc[i+2]['label']==3:
        if found_count == 0:
            print(f'\n✓ First triplet at rows {i}-{i+2}:')
            print(df.iloc[i:i+3][['label', 'block', 'transaction_index']].to_string())
        found_count += 1
        
print(f'\nFound {found_count} consecutive triplets in first 1000 rows')

# Split WITHOUT shuffling - preserve exact order
# 70% train, 15% val, 15% test
total = len(df)
train_end = int(0.70 * total)
val_end = int(0.85 * total)

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

print(f'\n--- Split (preserving exact order) ---')
print(f'Train: rows 0-{train_end-1} ({len(train_df):,} rows)')
print(f'Val:   rows {train_end}-{val_end-1} ({len(val_df):,} rows)')
print(f'Test:  rows {val_end}-{total-1} ({len(test_df):,} rows)')

# Save
train_df.to_csv('../SandwichDetection/data/train.csv', index=False)
val_df.to_csv('../SandwichDetection/data/val.csv', index=False)
test_df.to_csv('../SandwichDetection/data/test.csv', index=False)

print('\n✓ Saved to SandwichDetection/data/')
print('  - train.csv')
print('  - val.csv')
print('  - test.csv')
print('\nOrder preserved - NO SHUFFLING!')
