# =========================================
# Build a Linear Regression Model from Scratch with NumPy
# =========================================
import numpy as np


class LinearRegression:
    def __init__(self, lr=0.01, epochs=2000):
        # Learning rate — controls the step size of gradient descent
        # 学习率 — 控制梯度下降每次更新的步幅
        self.lr = lr

        # Number of training iterations (epochs)
        # 训练轮数（迭代次数）
        self.epochs = epochs

        # Weight coefficient (slope)
        # 权重参数（斜率）
        self.w = None

        # Bias term (intercept)
        # 偏置参数（截距）
        self.b = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        # Initialize model parameters
        # 初始化模型参数
        self.w = 0.0
        self.b = 0.0

        m = len(X)  # Number of training samples 样本数量

        # Gradient Descent Optimization 梯度下降优化
        for _ in range(self.epochs):
            # Forward propagation (prediction)
            # 前向传播（预测）
            y_pred = self.w * X + self.b

            # Compute gradients for w and b using the Mean Squared Error (MSE) loss
            # 使用均方误差（MSE）损失函数计算参数 w 和 b 的梯度
            dw = (2 / m) * np.sum((y_pred - y) * X)
            db = (2 / m) * np.sum(y_pred - y)

            # Parameter update rule
            # 参数更新公式
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        # Make predictions on new data
        # 在新数据上进行预测
        X = np.array(X)
        return self.w * X + self.b


# =========================================
# Example Usage / 示例代码
# =========================================

# Generate synthetic linear data with noise
# 人工生成线性数据（含噪声），用于模拟真实场景
X = np.linspace(0, 10, 50)
y = 2 * X + 6 + np.random.randn(50)  # true relation: y = 2x + 6 + noise

# Initialize and train the model
# 定义并训练模型
model = LinearRegression(lr=0.01, epochs=2000)
model.fit(X, y)

print("Trained weight (w):", round(model.w, 3))
print("Trained bias (b):", round(model.b, 3))

# Test prediction
# 模型预测
test_x = np.array([1.2, 3.4, 5.6, 7.8])
pred_y = model.predict(test_x)

print(f"When x = {test_x}, predicted y = {np.round(pred_y, 2)}")
