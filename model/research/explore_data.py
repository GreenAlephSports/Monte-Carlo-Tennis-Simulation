from data_loader import load_matches

df = load_matches()

print("Columns:")
print(df.columns.tolist())

print("\nFirst 10 rows:")
print(df.head(10))
