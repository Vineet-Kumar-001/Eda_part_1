import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------------------------------------------------
# 📌 Create folder for saving PNGs
# ---------------------------------------------------------
output_path = os.path.join(os.getcwd(), "plots")
os.makedirs(output_path, exist_ok=True)

# ---------------------------------------------------------
# 📌 Load dataset
# ---------------------------------------------------------
cars = pd.read_csv(r"Location of file copy the path ")

print("Shape:", cars.shape)
print("\nData Info:")
print(cars.info())
print("\nColumns:", cars.columns.tolist())

print("\nHead:")
print(cars.head())
print("\nRandom Sample:")
print(cars.sample(5))

# ---------------------------------------------------------
# 🧹 Clean Data
# ---------------------------------------------------------
cars = cars.drop(['MSRP', 'Invoice'], axis=1)

cars = cars.drop_duplicates(keep='first')

print("\nMissing values:")
print(cars.isnull().sum())

cars.dropna(inplace=True)

# ---------------------------------------------------------
# 🔽 Sort Data
# ---------------------------------------------------------
cars_sort = cars.sort_values(by='MPG_City', ascending=False)
print("\nSorted Data:")
print(cars_sort.head())

# ---------------------------------------------------------
# 🌍 Origin Stats
# ---------------------------------------------------------
print("\nContinent Counts:")
print(cars['Origin'].value_counts())

print("\nUnique Car Makes:")
print(cars['Make'].unique())

# ---------------------------------------------------------
# 🔢 Encode Origin
# ---------------------------------------------------------
def origin_to_num(x):
    if x == "Asia": return 1
    if x == "Europe": return 2
    if x == "USA": return 3

cars['Origin'] = cars['Origin'].apply(origin_to_num)

# ---------------------------------------------------------
# 🔎 iloc/loc Examples
# ---------------------------------------------------------
print("\nFirst Row:", cars.iloc[0])
print("\nLast Row:", cars.iloc[-1])
print("\nFirst 5 rows using iloc:")
print(cars.iloc[0:5])
print("\nColumn 0:")
print(cars.iloc[:, 0])
print("\nRows 0,2,4 — Columns 1,3,5:")
print(cars.iloc[[0, 2, 4], [1, 3, 5]])

x = cars.loc[0:5, 'MPG_City']
print("\nSelected Series:", x)

# ---------------------------------------------------------
# 🔢 Select Numerical Columns
# ---------------------------------------------------------
cars1 = cars[['EngineSize', 'Cylinders', 'Horsepower', 'MPG_City',
              'Weight', 'Wheelbase', 'Length']]

cars = cars.select_dtypes(include=['float64', 'int64'])

# ---------------------------------------------------------
# 📊 Summary & Correlation
# ---------------------------------------------------------
print("\nSummary Statistics:")
print(cars.describe())

print("\nCorrelation with MPG_City:")
print(cars.corr()['MPG_City'])

# ---------------------------------------------------------
# 📈 Graphs — Save every graph to PNG
# ---------------------------------------------------------

# 1. Distribution Plot
plt.figure(figsize=(7, 5))
sns.displot(data=cars, x='MPG_City', bins=10, kde=True, color='blue')
plt.savefig(os.path.join(output_path, "mpg_distribution.png"))
plt.close()

# 2. Regression Plot
plt.figure(figsize=(7, 5))
sns.regplot(x='Length', y='MPG_City', data=cars)
plt.savefig(os.path.join(output_path, "length_vs_mpg_regression.png"))
plt.close()

# 3. Boxplot
plt.figure(figsize=(7, 5))
sns.boxplot(x='Origin', y='MPG_City', data=cars)
plt.savefig(os.path.join(output_path, "origin_mpg_boxplot.png"))
plt.close()

# 4. Pairplot
sns.pairplot(cars, hue='Origin')
plt.savefig(os.path.join(output_path, "pairplot.png"))
plt.close()

# 5. Heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(cars.corr(), annot=True, cmap="coolwarm")
plt.savefig(os.path.join(output_path, "correlation_heatmap.png"))
plt.close()

print("\n🎉 All graphs saved in folder:", output_path)
