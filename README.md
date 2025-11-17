# Car Dataset Analysis

This project performs data cleaning, analysis, visualization, and feature exploration on a Car Dataset using Python. It also generates multiple visualizations and automatically saves them inside a **plots/** folder.

---

## 🚀 Features

* Load and inspect dataset (shape, info, columns, head, random samples)
* Data cleaning (drop columns, remove duplicates, remove missing values)
* Sorting and exploring categorical features
* Encoding categorical column (`Origin`)
* Using `iloc` and `loc` for powerful indexing
* Selecting numerical features for analysis
* Summary statistics and correlation matrix
* Multiple graphs saved as PNG

---

## 📂 Folder Structure

```
project-folder/
│
├── plots/                # Saved PNG graphs
│   ├── mpg_distribution.png
│   ├── length_vs_mpg_regression.png
│   ├── origin_mpg_boxplot.png
│   ├── pairplot.png
│   └── correlation_heatmap.png
│
└── eda.py        # Main analysis code
```

---

## 📦 Requirements

Install dependencies:

```bash
pip install pandas matplotlib seaborn
```

---

## ▶️ How to Run the Code

### **📌 Run Locally in VS Code / PyCharm**

```bash
eda.py
```

All graphs will appear in your `plots/` folder.

---

## ▶️ Run on Google Colab

You can directly upload your dataset and run the script.

### **Step 1: Upload your CSV file**

```python
from google.colab import files
uploaded = files.upload()
```

### **Step 2: Run the code**

Copy the entire Python script into a Colab cell and replace:

```python
cars = pd.read_csv("your_file_location.csv")
```

with:

```python
cars = pd.read_csv(list(uploaded.keys())[0])
```

### **Step 3: View saved plots**

Plots will be saved in:

```
/content/plots/
```

To display them inside Colab:

```python
from IPython.display import Image
Image('/content/plots/mpg_distribution.png')
```

---

## 📈 Visualizations Generated

This project generates:

* MPG distribution plot
* Length vs MPG regression plot
* Origin vs MPG boxplot
* Pairplot of all numeric features
* Heatmap of correlations

All saved inside **plots/** automatically.

---

## 🧠 Purpose

This analysis helps understand:

* Relationship between car features
* How different origins affect performance
* Patterns in mileage, horsepower, weight, and dimensions

Perfect for Data Science beginners and ML students.

---


