# pyfit-ultime

This library is designed to reproduce machine learning and deep learning algorithms from scratch in order to understand their subtleties.


<!----------------------------------------------------------------------------->

# 1. Machine Learning functions and classes

<!----------------------------------------------------------------------------->

## 1.1. Data processing

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

## 1.2. Loss Functions

### pyfit.loss.mean_squared_error

Compute MSE between y_true ant y_pred.

### pyfit.loss.mean_absolute_error

Compute MAE between y_true ant y_pred.

### pyfit.loss.log_loss

Compute LogLoss between y_true ant y_pred.

<!----------------------------------------------------------------------------->

## 1.3. Metrics

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

## 1.4. Machine Learning Models

### pyfit.linear_reg.LinearReg

<!-- TODO-->

### pyfit.logistic_reg.LogisticReg

<!-- TODO-->

### pyfit.kmeans.KMeans

```python
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)           # Compute k-means clustering
print(kmeans.centers_)
print(kmeans.labels_)
Y = kmeans.predict(X)   # Predict the closest center each sample in X belongs to.
```

### pyfit.neighbors.KNeighborsClassifier

<!-- TODO-->

### pyfit.neighbors.KNeighborsRegressor

<!-- TODO-->

### pyfit.decision_tree_classifier.DecisionTreeClassifier

A decision tree classifier, parameters:
- criterion: 'gini' only
- max_depth: bound for the tree depth, None for unlimited

```python
clf = DecisionTreeClassifier(criterion='gini', max_depth=5)
clf.fit(X, Y)
y_pred = clf.predict(X)
```

<!----------------------------------------------------------------------------->
<!----------------------------------------------------------------------------->

# 2. Deep Learning functions and classes

<!----------------------------------------------------------------------------->

## 2.1 Engine - Autograd

Deep Learning is made through automatic differentiation. You must use Tensor for your dataset.

### pyfit.engine.Tensor

Stores values and their gradients. Shape must be (batch_size, nb_features).
Input with shape () and (n,) will be convert to (1, n). A tensor is always 2D array!

Supported operations:

```python
x.zero_grad()   # reset gradient to zero
z = x.mean()    # input_shape: (m,n) --> output_shape: (m,1)
z = x.abs()
z = x.relu()
z = x.exp()
z = x.log()
z = x + y       # only when shapes can be broadcast according to numpy
z = x + 1       # equivalent to z = 1 + x
z = x - 1       # equivalent to z = x + (-1)
z = 1 - x       # equivalent to z = 1 + (-x)
z = -x
z = x * y       # only when shapes are identical
z = x * 2       # allowed with scalar but no gradient for the scalar
z = 2 * x       # allowed with scalar but no gradient for the scalar
z = x / y       # only when shapes are identical
z = x / 2       # allowed with scalar but no gradient for the scalar
z = 1 / x       # allowed with scalar but no gradient for the scalar
z = x ** 2      # only with integer exponent
z = x.dot(y)    # matrix product
```

Calling z.backward() after operations will automatically compute all gradients for all tensor involved in the computation of z.

### pyfit.engine.as_tensor

Convert scalar, list or np.ndarray to Tensor if necessary.

<!----------------------------------------------------------------------------->

## Activation functions

All of this functions can be used with numpy.ndarray and Tensor.

### pyfit.activation.sigmoid

Compute sigmoid function on x: 1 / (1+exp(-x)).

### pyfit.activation.relu

Compute ReLU function on x: max(x, 0)

### pyfit.activation.tanh

Compute tanh function on x using numpy.

<!----------------------------------------------------------------------------->

## 2.2 Neural Networks

For all layers, at least input_length is necessary in order to build a sequential model.

### pyfit.nn.Neuron

Compute a single Neuron. The calculation is the sum of its inputs plus a bias. Then the result is activated by an activation function.

```python
n = Neuron(in_features=3, activation='sigmoid')
x = Tensor([1,1,1])
y = n(x)
print(y.shape)  # (1,1)
parameters = n.parameters() # list of tensors [weights, bias]
```

### pyfit.nn.Activation

Apply an activation function to its inputs. Input length is necessary.
```python
act = Activation(in_features=3, activation='relu')
```

### pyfit.nn.Dense

Same as Neuron but with a layer of Neurons. Input length and output length are necessary.
```python
fully_connected_layer = Dense(in_features=3, out_features=2, activation='linear')
```

### pyfit.nn.Dropout

The Dropout layer deactivate some neurons during training with probability based on the rate parameter. Input length is necessary.
```python
dropout_layer = Dropout(in_features=3, rate=0.4)
```

### pyfit.nn.Sequential

Stack several layers in one model.

```python
model = Sequential()
model.add(Dense(2, 4, activation='relu'))
model.add(Dense(4, 1, activation='sigmoid'))
# forward pass
y_pred = model(X)
parameters = model.parameters() # list of all parameters of all layers (flatten).
```

### pyfit.optim.SGD

Only optimizer implemented, Stochastic Gradient Descent updates model parameters in the opposite direction of their gradient.

```python
opt = SGD(model.parameters())
y = model(x)
z = some_loss_function(y)
z.backward()
opt.step()  # updates weights and bias in model using gradients
```

### pyfit.trainer.Trainer

```python
trainer = Trainer(model, optimizer, loss)
history = trainer.fit(dataset, num_epochs=200, verbose=True)
```
history will be a dict containing loss and binary_accuracy for each epoch.

Here is the main training loop:

```python
model = ...
opt = SGD(model.parameters(), learning_rate=1e-2)

for epoch in range(n_epochs):
    opt.zero_grad()
    for batch in dataset:   # using BatchIterator
        y_pred = model(batch.inputs)
        loss = some_loss_function(batch.targets, y_pred)
        loss.backward()
        opt.step()
```
