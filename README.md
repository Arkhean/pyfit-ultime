# pyfit-ultime

# Machine Learning functions and classes

<!----------------------------------------------------------------------------->

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

<!----------------------------------------------------------------------------->

## Loss Functions

### pyfit.loss.mean_squared_error

Compute MSE between y_true ant y_pred.

### pyfit.loss.mean_absolute_error

Compute MAE between y_true ant y_pred.

### pyfit.loss.log_loss

Compute LogLoss between y_true ant y_pred.

<!----------------------------------------------------------------------------->

## Metrics

### pyfit.metrics.euclidean_distance

Euclidean distance: https://en.wikipedia.org/wiki/Euclidean_distance.

### pyfit.metrics.accuracy_score

Compute accuracy classification between y_true and y_pred.
Accuracy is defined as #TruePositive / #All.

### pyfit.metrics.recall_score

Compute recall between y_true and y_pred.
Recall is defined as #TruePositive / (#TruePositive + #FalsePositive)

### pyfit.metrics.precision_score

Compute precision between y_true and y_pred.
Precision is defined as #TruePositive / (#TruePositive + #FalseNegative)

### pyfit.metrics.binary_accuracy

<!----------------------------------------------------------------------------->

## Machine Learning Models

### pyfit.linear_reg.LinearReg

### pyfit.logistic_reg.LogisticReg

### pyfit.kmeans.KMeans

### pyfit.neighbors.KNeighborsClassifier

### pyfit.neighbors.KNeighborsRegressor

### pyfit.decision_tree_classifier.DecisionTreeClassifier

<!----------------------------------------------------------------------------->
<!----------------------------------------------------------------------------->

# Deep Learning functions and classes

<!----------------------------------------------------------------------------->

## Engine - Autograd

### pyfit.engine.Tensor

Stores values and their gradients. Shape must be (batch_size, nb_features).
Input with shape () and (n,) will be convert to (1, n). A tensor is always 2D array!

Supported operations:

```python
x.zero_grad()
z = x.mean()    # input_shape: (m,n) --> output_shape: (m,1)
z = x.abs()
z = x.relu()
z = x.exp()
z = x.log()
z = x + y       # only when shapes can be broadcast according to numpy
z = x + 1       # equivalent to z = 1 + x
z = x - 1       # equivalent to z = -(1 - x)
z = -x
z = x * y       # only when shapes are identical
z = x * 2       # allowed with scalar but no gradient for the scalar
z = 2 * x       # allowed with scalar but no gradient for the scalar
z = x / y       # only when shapes are identical
z = x / 2       # allowed with scalar but no gradient for the scalar
z = 1 / x       # allowed with scalar but no gradient for the scalar
z = x ** 2      # only with integer exponent
z = x.dot(y)
```

Calling z.backward() after operations will automatically compute all gradients for all tensor involved in the computation of z.

### pyfit.engine.as_tensor

Convert scalar, list or np.ndarray to Tensor if necessary.

<!----------------------------------------------------------------------------->

## Activation functions

### pyfit.activation.sigmoid

### pyfit.activation.relu

### pyfit.activation.tanh

<!----------------------------------------------------------------------------->

## Neural Networks

### pyfit.nn.Neuron

### pyfit.nn.Activation

### pyfit.nn.Dense

### pyfit.nn.Dropout

### pyfit.nn.Sequential

### pyfit.optim.Optimizer

### pyfit.trainer.Trainer
