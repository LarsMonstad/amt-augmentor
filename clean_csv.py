import pandas as pd

# Read the CSV file
df = pd.read_csv('HF32025.csv')

# Function to check if a file is augmented
def is_augmented(filename):
    augmentations = ['timestretch', 'pitchshift', 'reverb_filters', 'gain_chorus', 'addpauses']
    return any(aug in filename for aug in augmentations)

# Keep train split as is, remove augmented files from test and validation
cleaned_df = df[
    (df['split'] == 'train') |  # keep all training data
    ((df['split'] != 'train') & (~df['audio_filename'].apply(is_augmented)))  # keep only original files for test/val
]

# Save the cleaned dataset
cleaned_df.to_csv('HF32025_cleaned.csv', index=False)

# Print statistics
print("\nCleaned dataset distribution:")
print(cleaned_df['split'].value_counts())
print(f"\nTotal files: {len(cleaned_df)}")
