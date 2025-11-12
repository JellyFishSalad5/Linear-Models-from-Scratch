# =========================================
# Build a Multivariate Linear Regression Model from Scratch with NumPy
# 使用 NumPy 手动实现多维线性回归模型
# =========================================

import numpy as np

class LinearRegressionMulti:
    def __init__(self, lr=0.01, epochs=2000):
        # Learning rate — controls the step size in gradient descent
        # 学习率 — 控制梯度下降更新步幅
        self.lr = lr

        # Number of training iterations (epochs)
        # 训练轮数（迭代次数）
        self.epochs = epochs

        # Model parameters: weights and bias
        # 模型参数：权重和偏置
        self.w = None
        self.b = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        # m: number of samples; n: number of features
        # m 表示样本数量；n 表示特征维度
        m, n = X.shape

        # Initialize parameters
        # 初始化参数
        self.w = np.zeros(n)
        self.b = 0.0

        # Gradient Descent Optimization 梯度下降优化
        for _ in range(self.epochs):
            # Forward propagation (prediction)
            # 前向传播（预测）
            y_pred = X.dot(self.w) + self.b

            # Compute gradients based on Mean Squared Error (MSE)
            # 基于均方误差（MSE）计算梯度
            dw = (2 / m) * X.T.dot(y_pred - y)   # Gradient for weights 权重梯度
            db = (2 / m) * np.sum(y_pred - y)    # Gradient for bias 偏置梯度

            # Update parameters
            # 更新参数
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        # Generate predictions for new input data
        # 对新输入数据进行预测
        X = np.array(X)
        return X.dot(self.w) + self.b


# =========================================
# Example Usage / 示例代码
# =========================================

# Generate synthetic multivariate data: y = 2*x1 + 3*x2 + 5 + noise
# 生成模拟数据：y = 2*x1 + 3*x2 + 5 + 噪声
np.random.seed(0)
m = 100
X = np.random.rand(m, 2)
y = 2 * X[:, 0] + 3 * X[:, 1] + 5 + np.random.randn(m) * 0.1

# Initialize and train the model
# 初始化并训练模型
model = LinearRegressionMulti(lr=0.1, epochs=5000)
model.fit(X, y)

print("Trained weights (w):", round(model.w, 3))
print("Trained bias (b):", round(model.b, 3))

# Test predictions
# 模型预测
test_X = np.array([
    [0.5, 0.5],
    [1.0, 2.0],
    [0.1, 0.9]
])
pred_y = model.predict(test_X)

print("Predicted results:", np.round(pred_y, 3))
