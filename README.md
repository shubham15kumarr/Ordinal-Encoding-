# Ordinal Encoding and Preprocessing Pipeline

This project demonstrates how to preprocess a dataset using Scikit-Learn's `ColumnTransformer` by applying different encoding and imputation techniques, including **Ordinal Encoding**, **OneHot Encoding**, and **Missing Value Imputation**.

---

## 📌 Objective

To apply different preprocessing methods on a dataset and prepare it for machine learning models:
- Handle missing values in numerical data.
- Apply ordinal encoding on categorical features with inherent order.
- Apply one-hot encoding on nominal categorical features.

---

## 🛠️ Technologies & Libraries Used

- Python 3
- NumPy
- Pandas
- Scikit-learn (SimpleImputer, OrdinalEncoder, OneHotEncoder, ColumnTransformer)

---

## 🔄 Preprocessing Steps

1. **SimpleImputer** applied to handle missing values in the `fever` column.
2. **OrdinalEncoder** applied on the `cough` column (categories: `['poor', 'average','good]`['school','UG','PG']).
3. **OneHotEncoder** applied to `gender` and `city` columns (with `drop='first'` to avoid dummy variable trap).
4. Combined using `ColumnTransformer` with `remainder='passthrough'` to retain other columns.(we can also used)

---

## 📊 Output

The final transformed dataset is a NumPy array combining:
- Imputed values
- Encoded values
- Remaining numerical features

```python
print(X_train_transformed.shape)  # Output shape after preprocessing
