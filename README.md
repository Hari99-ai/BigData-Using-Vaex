# BigData-Using-Vaex

This repository demonstrates how to work with **large datasets** efficiently using **Vaex**, a high-performance Python library for out-of-core DataFrames. It includes data handling, preprocessing, and machine learning using Vaex and scikit-learn.

---

## 📂 Repository Files

- `vaex.ipynb` – Jupyter notebook demonstrating data loading, processing, and ML with Vaex.  
- `large_data_exported.hdf5` – Sample HDF5 dataset for testing Vaex operations.

---

## 🚀 Features

1. **Load Large HDF5 Files**  
   Load huge datasets efficiently using `vaex.open()` without loading everything into memory.  

2. **Data Conversion**  
   Convert Pandas DataFrames to Vaex for faster processing:
  

3. **Shuffling & Train-Test Split**
   Shuffle data and split into training and test sets efficiently:

   ```python
   vaex_df = vaex_df.shuffle()
   df_train, df_test = vaex_df.ml.train_test_split(test_size=0.2)
   ```

4. **Incremental Machine Learning**
   Train models on large datasets incrementally using Vaex + scikit-learn:

   ```python
   from vaex.ml.sklearn import IncrementalPredictor
   from sklearn.linear_model import SGDRegressor

   features = ['f1','f2','f3','f4']
   target = 'target'
   model = SGDRegressor()
   vaex_model = IncrementalPredictor(features=features, target=target, model=model, batch_size=500000)
   vaex_model.fit(df=df_train)
   df_test = vaex_model.transform(df_test)
   ```

5. **Evaluation**
   Evaluate regression performance on large datasets:

   ```python
   from sklearn.metrics import r2_score, mean_absolute_error

   r2 = r2_score(df_test['target'].values, df_test['prediction'].values)
   mae = mean_absolute_error(df_test['target'].values, df_test['prediction'].values)
   print("R2 Score:", r2)
   print("MAE:", mae)
   ```

---

## ⚡ Installation

Install Vaex with all dependencies:

```bash
pip install "vaex[full]"
```

---

## 📊 Example Output

After training an incremental regression model, a test dataset prediction may look like:

| f1        | f2        | f3        | f4       | target   | prediction |
| --------- | --------- | --------- | -------- | -------- | ---------- |
| 0.276644  | -0.0668   | 0.340587  | 0.624418 | -25.8545 | -1.78298   |
| -0.434699 | -0.365077 | -0.378762 | 0.803824 | -7.42881 | -9.40138   |
| ...       | ...       | ...       | ...      | ...      | ...        |

R2 Score: 0.591
MAE: 15.967

---

## 📖 References

* [Vaex Documentation](https://vaex.io/docs/)
* [HDF5 Format](https://www.hdfgroup.org/solutions/hdf5/)
* [scikit-learn Incremental Learning](https://scikit-learn.org/stable/modules/scaling_strategies.html#incremental-learning)

---

## 💡 Notes

* Vaex is optimized for **memory efficiency** and **large datasets**.
* Always shuffle your data before splitting for training/testing.
* Incremental predictors allow handling datasets that **cannot fit into memory**.

---

## Author

**Hari Om** – [GitHub](https://github.com/Hari99-ai)

```

---

If you want, I can also make a **more concise “GitHub-friendly” version** that looks clean with badges and quick setup instructions.  

Do you want me to do that?
```
