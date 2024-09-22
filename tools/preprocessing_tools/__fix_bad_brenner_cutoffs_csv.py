import pandas as pd
import os

path = "/home/labs/hornsteinlab/Collaboration/MOmaps/src/preprocessing/sites_validity_bounds_spd18days.csv"
print(f"Loading {path}")
df = pd.read_csv(path, index_col=0)
df['Site_brenner_lower_bound'] = df[df.columns[-1]]
df = df.drop(columns=df.columns[-1])

filename, ext = os.path.splitext(path)
savepath = f"{filename}_fixed{ext}"
df.to_csv(savepath)
print(f"Fixed file was saved to {savepath}")