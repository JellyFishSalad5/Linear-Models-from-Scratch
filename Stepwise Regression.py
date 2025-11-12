# =========================================
# Stepwise Linear Regression (Forward Selection) implemented with NumPy
# 前向逐步回归（Stepwise Regression）—— 使用 NumPy 从零实现
# =========================================

import numpy as np

class StepwiseLinearRegression:
    def __init__(self, lr=0.01, epochs=2000, max_features=None):
        # Learning rate — controls the step size in gradient descent
        # 学习率 — 控制梯度下降的更新步幅
        self.lr = lr

        # Number of training iterations (epochs)
        # 训练轮数（迭代次数）
        self.epochs = epochs

        # Maximum number of features to select (None -> all)
        # 最大选择特征数（None 表示可选所有特征）
        self.max_features = max_features

        # Selected feature indices after fitting
        # 训练后被选中的特征索引
        self.selected_features = []

        # Model parameters: weights and bias
        # 模型参数：权重向量和偏置
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        Fit the model using forward stepwise selection based on MSE.
        使用基于 MSE 的前向逐步特征选择训练模型。
        """
        X = np.array(X)
        y = np.array(y)
        m, n = X.shape  # m: number of samples, n: number of features

        if self.max_features is None:
            self.max_features = n

        remaining_features = list(range(n))
        self.selected_features = []
        self.w = np.array([])  # will hold weights corresponding to selected features
        self.b = 0.0

        # Forward selection loop: iteratively add the feature that yields lowest MSE
        # 前向选择主循环：每次尝试加入能最小化 MSE 的特征
        for _ in range(self.max_features):
            best_mse = float('inf')
            best_feature = None
            best_w, best_b = None, None

            for f in remaining_features:
                # candidate feature set = already selected features + current candidate f
                current_features = self.selected_features + [f]
                X_sub = X[:, current_features]  # submatrix with selected features

                # Initialize submodel parameters for this candidate
                w = np.zeros(len(current_features))
                b = 0.0

                # Train submodel with gradient descent using MSE loss
                # 使用梯度下降训练子模型（均方误差为损失函数）
                for _ in range(self.epochs):
                    y_pred = X_sub.dot(w) + b
                    dw = (2 / m) * X_sub.T.dot(y_pred - y)  # gradient wrt weights
                    db = (2 / m) * np.sum(y_pred - y)       # gradient wrt bias
                    w -= self.lr * dw
                    b -= self.lr * db

                # Evaluate candidate model by MSE
                mse = np.mean((X_sub.dot(w) + b - y) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    best_feature = f
                    best_w = w.copy()
                    best_b = b

            # If a best feature was found, add it into selected list and update params
            if best_feature is not None:
                self.selected_features.append(best_feature)
                remaining_features.remove(best_feature)
                self.w = best_w
                self.b = best_b
            else:
                # No improvement possible
                break

    def predict(self, X):
        """
        Predict using the selected features.
        使用被选中的特征进行预测。
        """
        X = np.array(X)
        if len(self.selected_features) == 0:
            # If no features selected, return constant prediction = bias
            return np.full((X.shape[0],), self.b)
        return X[:, self.selected_features].dot(self.w) + self.b


# =========================================
# Example usage (demo) / 示例运行
# =========================================
if __name__ == "__main__":
    np.random.seed(0)
    m = 100
    # Generate synthetic data: only first 3 features actually contribute to y
    X = np.random.rand(m, 5)
    y = 2 * X[:, 0] + 3 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(m) * 0.1

    # Forward selection: choose up to 3 features
    model = StepwiseLinearRegression(lr=0.1, epochs=3000, max_features=3)
    model.fit(X, y)

    # Unified, rounded training summary (3 decimal places)
    print("\n=== Training Summary ===")
    print("Selected features:", model.selected_features)
    print("Weights (w):", np.round(model.w, 3))
    print("Bias (b):", round(model.b, 3))
    print("=========================")

    # Test predictions
    test_X = np.random.rand(3, 5)
    pred_y = model.predict(test_X)
    print("Predicted values:", np.round(pred_y, 3))
