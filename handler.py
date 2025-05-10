import runpod
from train import train_model

def handler(job):
    input_data = job["input"]
    epochs = input_data.get("epochs", 10)
    lr = input_data.get("lr", 0.01)
    result = train_model(epochs=epochs, lr=lr)
    return result

runpod.serverless.start({"handler": handler})
