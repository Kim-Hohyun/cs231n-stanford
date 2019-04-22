import numpy as np

def svm_loss_vectorized(W, X, y, reg):

    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    num_classes = W.shape[1]
    num_train = X.shape[0]
    
    # compute the loss 
    Z = np.dot(X, W)
    correct_score = Z[np.arange(num_train), y].reshape(-1,1)
    Z = Z - correct_score + 1 
    Z[np.arange(num_train), y] = 0
    loss = (np.sum(Z[Z>0]) / num_train) + (reg * np.sum(W * W))

    # compute gradients
    Z = np.where(Z>0, 1, 0)
    Z[np.arange(num_train), y] = -np.sum(Z, axis=1)
    dW += np.dot(X.T, Z) / num_train
    dW += (2 * reg * W)
    
    return loss, dW

class LinearSVM():
    def __init__(self):
        self.W = None
    
    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):

        num_train, dim = X.shape
        num_classes = np.max(y) + 1
        
        if self.W is None: # 훈련을 이어서 하는 것이 가능하도록 만든다.
            self.W = 0.001 * np.random.randn(dim, num_classes)
        
        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):

            batch_index = np.random.choice(np.arange(num_train), batch_size)
            X_batch = X[batch_index]
            y_batch = y[batch_index]

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # perform parameter update
            self.W -= learning_rate * grad

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        
        y_pred = np.argmax(np.dot(X, self.W), axis=1)
        
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)

class TwoLayerNet(): # input - fully connected layer - ReLU - fully connected layer - softmax
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):

        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N = X.shape[0]
        
        # Compute the forward pass
        Z1 = np.dot(X, W1) + b1
        A1 = np.maximum(0, Z1)
        Z2 = np.dot(A1, W2) + b2       
        if y is None:
            return Z2

        # Compute the loss
        scores_exp = np.exp(Z2)
        scores_expsum = np.sum(scores_exp, axis=1, keepdims=True)
        loss = -np.sum(Z2[range(N), y])
        loss += np.sum(np.log(scores_expsum))
        loss /= N
        loss += reg * (np.sum(W1*W1) + np.sum(W2*W2))
        
        # Backward pass: compute gradients
        grads = {}
        dZ2 = np.zeros(Z2.shape)
        dZ2[range(N), y] = -1
        dZ2 += scores_exp / scores_expsum
        grads['W2'] = np.dot(A1.T, dZ2) / N + 2 * reg * W2
        grads['b2'] = np.sum(dZ2, axis=0) / N
        
        dA1 = np.dot(dZ2, W2.T)
        dA1[Z1<=0] = 0
        dZ1 = dA1
        grads['W1'] = np.dot(X.T, dZ1) / N + 2 * reg * W1
        grads['b1'] = np.sum(dZ1, axis=0) / N

        return loss, grads
    
    def train(self, X, y, X_val, y_val, learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100, batch_size=200, verbose=False):
        
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            batch_index = np.random.choice(range(num_train), batch_size)
            X_batch = X[batch_index]
            y_batch = y[batch_index]

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)
            
            # SGD
            self.params['W1'] -= learning_rate * grads['W1']
            self.params['b1'] -= learning_rate * grads['b1']
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b2'] -= learning_rate * grads['b2']
            
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))
            
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        scores = self.loss(X)
        y_pred = np.argmax(scores, axis=1)
        return y_pred