# Sample retrieval (Part 2)

Query: `What is the difference between bagging and boosting?`

## Rank 1 - score 0.0816 - intro-to-ml-book.pdf, page 103 - chunk `intro-to-ml-book.pdf::0067`

> in the ensemble, another impor‐
> tant parameter of gradient boosting is the learning_rate, which controls how
> strongly each tree tries to correct the mistakes of the previous trees. A higher learning
> rate means each tree can make stronger corrections, allowing for more complex mod‐
> els. Adding more trees to the ensemble, which can be accomplished by increasing
> n_estimators, also increases the model complexity, as the model has more chances
> to correct mistakes on the training set.
> Here is an example of using GradientBoostingClassifier on the Breast Cancer
> dataset. By default, 100 trees of maximum depth 3 and a learning rate of 0.1 are used:
> In[72]:
> from sklearn.ensemble import GradientBoostingClassifier
> X_train, X_test, y_train, y_test = train_test_split(
> cancer.data, cancer.target, random_s

## Rank 2 - score 0.0754 - intro-to-ml-book.pdf, page 105 - chunk `intro-to-ml-book.pdf::0068`

> common approach is to first try random forests, which work quite robustly. If ran‐
> dom forests work well but prediction time is at a premium, or it is important to
> squeeze out the last percentage of accuracy from the machine learning model, mov‐
> ing to gradient boosting often helps.
> If you want to apply gradient boosting to a large-scale problem, it might be worth
> looking into the xgboost package and its Python interface, which at the time of writ‐
> ing is faster (and sometimes easier to tune) than the scikit-learn implementation of
> gradient boosting on many datasets.
> Strengths, weaknesses, and parameters. Gradient boosted decision trees are among the
> most powerful and widely used models for supervised learning. Their main drawback
> is that they require careful tuning of the parameters and m

## Rank 3 - score 0.0705 - intro-to-ml-book.pdf, page 102 - chunk `intro-to-ml-book.pdf::0066`

> more appropriate.
> Random forests usually work well even on very large datasets, and training can easily
> be parallelized over many CPU cores within a powerful computer. However, random
> forests require more memory and are slower to train and to predict than linear mod‐
> els. If time and memory are important in an application, it might make sense to use a
> linear model instead.
> The important parameters to adjust are n_estimators, max_features, and possibly
> pre-pruning options like max_depth. For n_estimators, larger is always better. Aver‐
> aging more trees will yield a more robust ensemble by reducing overfitting. However,
> there are diminishing returns, and more trees need more memory and more time to
> train. A common rule of thumb is to build “as many as you have time/memory for. ”
> As described
