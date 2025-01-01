from flask import Flask, request, jsonify

import torch
import torch.nn as nn
import numpy as np

app = Flask(__name__)


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = LeNet5()
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu'), weights_only=True))
model.eval()


@app.route('/')
def index():
    return "Welcome to the LeNet-5 Prediction API!"


@app.route('/health', methods=['GET']) 
def health_check(): 
    return jsonify({'status': 'API is running'}), 200


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    inputs = np.array(data['features']).reshape(1, 1, 28, 28)
    inputs = torch.tensor(inputs, dtype=torch.float32)

    with torch.no_grad():
        output = model(inputs)
        prediction = torch.argmax(output, dim=1).item()

    return jsonify({'prediction': prediction})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

