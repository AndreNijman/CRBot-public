from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Deque, Iterable, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from crbot.config import MODELS_DIR


class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """Minimal DQN agent with experience replay and soft ε-greedy decay."""

    def __init__(self, state_size: int, action_size: int, lr: float = 1e-3) -> None:
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.memory: Deque[Tuple] = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.action_size = action_size

    def update_target_model(self) -> None:
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return int(q_values.argmax().item())

    def replay(self, batch_size: int) -> None:
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            state_tensor = torch.as_tensor(state, dtype=torch.float32)
            next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32)

            target = reward
            if not done:
                target += self.gamma * torch.max(self.target_model(next_state_tensor))

            q_values = self.model(state_tensor)
            expected = q_values.clone().detach()
            expected[action] = float(target)

            prediction = q_values[action]
            loss = self.criterion(prediction, expected[action])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, filename: str | Path) -> None:
        path = Path(filename)
        if not path.is_absolute():
            path = MODELS_DIR / path
        load_kwargs = {"map_location": "cpu"}
        try:
            state_dict = torch.load(path, weights_only=True, **load_kwargs)
        except TypeError:
            state_dict = torch.load(path, **load_kwargs)
        model_state = self.model.state_dict()
        filtered = {}
        skipped = []
        for key, value in state_dict.items():
            if key in model_state and model_state[key].shape == value.shape:
                filtered[key] = value
            else:
                skipped.append(key)
        if not filtered:
            print(f"[DQNAgent] No compatible parameters found in checkpoint {path.name}; starting fresh.")
            return
        updated_state = model_state.copy()
        updated_state.update(filtered)
        self.model.load_state_dict(updated_state)
        self.model.eval()
        self.target_model.load_state_dict(self.model.state_dict())
        if skipped:
            preview = ", ".join(skipped[:5])
            if len(skipped) > 5:
                preview += ", ..."
            print(f"[DQNAgent] Skipped incompatible params from {path.name}: {preview}")

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        if not path.is_absolute():
            path = MODELS_DIR / path
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        return path
