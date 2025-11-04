import os
import time
import torch
import glob
import json
from env import ClashRoyaleEnv
from dqn_agent import DQNAgent
from pynput import keyboard
from datetime import datetime
from Actions import Actions  # uses press_play_again_keyburst()

class KeyboardController:
    def __init__(self):
        self.should_exit = False
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        try:
            if key.char == 'q':
                print("\nShutdown requested - cleaning up...")
                self.should_exit = True
        except AttributeError:
            pass

    def is_exit_requested(self):
        return self.should_exit

def get_latest_model_path(models_dir="models"):
    model_files = glob.glob(os.path.join(models_dir, "model_*.pth"))
    if not model_files:
        return None
    model_files.sort()
    return model_files[-1]

def train():
    env = ClashRoyaleEnv()
    agent = DQNAgent(env.state_size, env.action_size)
    actions = Actions()

    os.makedirs("models", exist_ok=True)

    latest_model = get_latest_model_path("models")
    if latest_model:
        agent.load(os.path.basename(latest_model))
        meta_path = latest_model.replace("model_", "meta_").replace(".pth", ".json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
                agent.epsilon = meta.get("epsilon", 1.0)
            print(f"Epsilon loaded: {agent.epsilon}")

    controller = KeyboardController()
    episodes = 10000
    batch_size = 32

    # periodic keyburst timer
    KEYBURST_PERIOD = 10.0  # seconds
    next_keyburst_at = time.time() + KEYBURST_PERIOD

    for ep in range(episodes):
        if controller.is_exit_requested():
            print("Training interrupted by user.")
            break

        state = env.reset()
        print(f"Episode {ep + 1} starting. Epsilon: {agent.epsilon:.3f}")
        total_reward = 0
        done = False

        while not done:
            # periodic keyburst regardless of detection
            now = time.time()
            if now >= next_keyburst_at:
                try:
                    actions.press_play_again_keyburst()  # presses '1' a few times
                finally:
                    next_keyburst_at = now + KEYBURST_PERIOD

            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay(batch_size)
            state = next_state
            total_reward += reward

        # also press once after episode end, for safety
        try:
            actions.press_play_again_keyburst()
        except Exception as e:
            print(f"End-of-episode keyburst failed: {e}")

        print(f"Episode {ep + 1}: Total Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.3f}")

        if ep % 10 == 0:
            agent.update_target_model()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join("models", f"model_{timestamp}.pth")
            torch.save(agent.model.state_dict(), model_path)
            with open(os.path.join("models", f"meta_{timestamp}.json"), "w") as f:
                json.dump({"epsilon": agent.epsilon}, f)
            print(f"Model and epsilon saved to {model_path}")

if __name__ == "__main__":
    train()
