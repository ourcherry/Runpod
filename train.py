import torch
import torch.nn as nn
import numpy as np

# 간단한 선형 데이터 생성 y = 2x + 3
def generate_data(n_samples=100):
    X = np.random.rand(n_samples, 1) * 10
    y = 2 * X + 3 + np.random.randn(n_samples, 1) * 0.5  # 노이즈 추가
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# 모델 정의
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# 학습 함수
def train_model(epochs=10, lr=0.01):
    X, y = generate_data()
    model = LinearRegressionModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        outputs = model(X)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # 결과 반환
    learned_weight = model.linear.weight.item()
    learned_bias = model.linear.bias.item()
    return {
        "status": "training complete",
        "epochs": epochs,
        "weight": round(learned_weight, 2),
        "bias": round(learned_bias, 2)
    }
