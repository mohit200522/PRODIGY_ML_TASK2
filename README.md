# **PRODIGY INFOTECH_ML_TASK_02**

## 🧩 **Customer Segmentation using K-Means Clustering**

A machine learning project that groups customers into different segments based on features like **Age, Annual Income, and Spending Score** using the **K-Means Clustering** algorithm.

---

## 📌 **Project Overview**

This project implements **K-Means**, an unsupervised machine learning algorithm, to identify **distinct customer groups**.
It is part of the **Prodigy Infotech Machine Learning Internship – Task 02**.

### **The goal is to:**

* Load and explore the dataset
* Preprocess and prepare the features
* Use Elbow & Silhouette methods to find optimal clusters
* Apply K-Means clustering
* Analyze and visualize the customer segments
* Understand business insights from clusters

---

## 📂 **Dataset**

You can use the popular **Mall Customers dataset** or the dataset provided.

### **Important Columns Used:**

* **Age**
* **Annual Income (k$)**
* **Spending Score (1–100)**

---

## 🚀 **Technologies Used**

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-Learn

---

## 🧠 **Model Used**

### **K-Means Clustering**

A popular **unsupervised learning** algorithm that groups data points based on similarity.

### **Why K-Means?**

* Simple and efficient
* Works well for grouping customers
* Helps identify patterns in behaviour
* Great for marketing & business insights

---

## 🛠️ **Project Workflow**

1. Import libraries
2. Load & explore dataset
3. Clean and preprocess data
4. Choose features (Age, Income, Spending Score)
5. Use **Elbow Method** to find ideal K
6. Validate with **Silhouette Score**
7. Train K-Means with best K
8. Visualize clusters using:

   * Heatmap
   * Pairplot
   * Boxplots
9. Interpret customer segments

---

## 📈 **Results**

The model successfully identifies **5 customer groups**, such as:

* High Income – High Spending
* Low Income – High Spending
* High Income – Low Spending
* Young customers with varying spending levels
* Older customers with moderate spending

### **Visualizations include:**

* Elbow curve
* Silhouette scores
* Cluster heatmap
* Pairplots
* Boxplots of clusters

---

## 📊 **Example Code Snippet**

```python
from sklearn.cluster import KMeans
import pandas as pd

# Load data
df = pd.read_csv("Mall_Customers.csv")
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Model
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
kmeans.fit(X)

# Prediction
labels = kmeans.predict(X)
```

---

## 📦 **How to Run**

1. Clone the repository
2. Install requirements →

   ```
   pip install -r requirements.txt
   ```
3. Run the notebook or script
4. View the cluster visualizations

---

## 🤝 **Contributing**

Contributions, issues, and suggestions are always welcome!

---

## 🧑‍💻 **Author**

**Mohit Vishwakarma**
Machine Learning Intern – Prodigy Infotech

⭐ *If you like this project, don't forget to star the repository!* ⭐


