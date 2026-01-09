import torch
import torch.nn as nn
import torch.optim as optim
from flytekit import task, workflow, Resources, ImageSpec
from flytekit.types.directory import FlyteDirectory
import os

# 保持 ImageSpec 以確保環境有 torch
# 如果網路依然有問題，可以暫時註解掉 container_image 並改用本地執行
training_image = ImageSpec(
    name="real-cpu-train-2",
    packages=[
        "torch", 
        "flytekit", 
        "cachetools", # 強制加入以解決 ModuleNotFoundError
        "pandas"
    ],
    registry="docker.io/zhan860127" 
)

@task(
    requests=Resources(cpu="1", mem="2Gi"),
    container_image=training_image,
    cache=True,
    cache_version="real-v1"
)
def train_model(epochs: int) -> FlyteDirectory:
    """真實的 CPU 訓練任務：訓練一個簡單的線性回歸模型"""
    print(f"開始訓練，預計執行 {epochs} 個 Epochs...")
    
    # 1. 準備數據 (y = 2x + 1)
    X = torch.randn(100, 1)
    y = 2 * X + 1 + torch.randn(100, 1) * 0.1
    
    # 2. 定義模型
    model = nn.Linear(1, 1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 3. 訓練迴圈
    for epoch in range(epochs):
        inputs, targets = X, y
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # 4. 儲存模型
    working_dir = os.path.abspath("/tmp/model_save")
    os.makedirs(working_dir, exist_ok=True)
    model_path = os.path.join(working_dir, "linear_model.pth")
    torch.save(model.state_dict(), model_path)
    
    return FlyteDirectory(path=working_dir)

@task(container_image=training_image)
def validate_model(model_dir: FlyteDirectory) -> float:
    """驗證模型：讀取權重並計算 MSE"""
    model_dir.download()
    model_path = os.path.join(model_dir.path, "linear_model.pth")
    
    # 載入模型
    model = nn.Linear(1, 1)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    # 測試數據
    test_x = torch.tensor([[5.0]])
    prediction = model(test_x).item()
    print(f"對於 x=5 的預測結果: {prediction} (期待接近 11)")
    
    return prediction

@workflow
def real_training_workflow(epochs: int = 50) -> float:
    model_dir = train_model(epochs=epochs)
    result = validate_model(model_dir=model_dir)
    return result