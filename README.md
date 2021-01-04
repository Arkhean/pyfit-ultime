# pyfit-ultime

# Machine Learning functions and classes

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Data processing

### pyfit.preprocessing.Scaler

Standardize features by removing the mean and scaling to unit variance.

```python
scaler = Scaler()
scaler.fit(X)           # Compute the mean and std to be used for later scaling.
Y = scaler.transform(X) # Perform standardization by centering and scaling

```

### pyfit.data.preprocessing.OneHotEncoder

Transform categorical data to vector with one one and zeros.
```python
enc = OneHotEncoder()
enc.fit(X)              # Fit OneHotEncoder to X
Y = enc.transform(X)    # Transform X using one-hot encoding.
```

### pyfit.data.train_test_split

Split data between train and test set with ratio: test_size=0.2
```python
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
```

### pyfit.data.make_classification

Generate clusters of points normally distributed.

### pyfit.data.plot_data

Plot some 2D data.

### pyfit.data.BatchIterator

Iterates on data by batches

```python
dataset = BatchIterator(X, Y, batch_size=32, shuffle=True)
for batch in dataset:
    x = batch.inputs
    y = batch.targets
```

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Loss Functions

### pyfit.loss.mean_squared_error

Compute MSE between y_true ant y_pred.

### pyfit.loss.mean_absolute_error

Compute MAE between y_true ant y_pred.

### pyfit.loss.log_loss

Compute LogLoss between y_true ant y_pred.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Metrics

### pyfit.metrics.euclidean_distance

### pyfit.metrics.accuracy_score

### pyfit.metrics.recall_score

### pyfit.metrics.precision_score

### pyfit.metrics.binary_accuracy

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Machine Learning Models

### pyfit.linear_reg.LinearReg

### pyfit.logistic_reg.LogisticReg

### pyfit.kmeans.KMeans

### pyfit.neighbors.KNeighborsClassifier

### pyfit.neighbors.KNeighborsRegressor

### pyfit.decision_tree_classifier.DecisionTreeClassifier

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Deep Learning functions and classes

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Engine - Autograd

### pyfit.engine.Tensor

### pyfit.engine.as_tensor

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Activation functions

### pyfit.activation.sigmoid

### pyfit.activation.relu

### pyfit.activation.tanh

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Neural Networks

### pyfit.nn.Neuron

### pyfit.nn.Activation

### pyfit.nn.Dense

### pyfit.nn.Dropout

### pyfit.nn.Sequential

### pyfit.optim.Optimizer

### pyfit.trainer.Trainer
