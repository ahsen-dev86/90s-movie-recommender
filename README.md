
# 🎬 90s Movie Recommender System  

This project is part of the **Elevvo Pathways Machine Learning Internship**. It builds an interactive **movie recommender app** using the **MovieLens 100K dataset (1990–1998)**.  

The app provides movie recommendations using three approaches:  
- **Item-based Collaborative Filtering**  
- **Matrix Factorization (SVD)**  
- **Hybrid (Item-based + SVD)**  

---

## 🚀 Features
- Movie recommendations based on ratings and genres.  
- Multiple genre filtering.  
- Three recommendation methods to compare.  
- Interactive visualizations:  
  - Ratings distribution  
  - Movies per genre  
  - Movie similarity heatmap  

---

## ⚙️ Tech Stack
- **Python**  
- **Streamlit** (app interface)  
- **Pandas, NumPy** (data handling)  
- **Scikit-learn** (cosine similarity, SVD)  
- **Matplotlib, Seaborn** (visualizations)  

---

## 📂 Dataset
Uses the **MovieLens 100K dataset**:  
- `u.item` → Movie details and genres  
- `u.data` → User ratings  

[Download here](https://grouplens.org/datasets/movielens/100k/)  

---

📊 Sample Output

Top-N recommended movies by method

Ratings histogram

Genre distribution barplot

Movie similarity heatmap
