from Src.data_loader import download_prices, save_to_csv, load_from_csv

df = download_prices()
print("Downloaded:", df.shape)

save_to_csv(df)
print("Saved CSV")

df2 = load_from_csv()
print("Loaded:", df2.shape)
