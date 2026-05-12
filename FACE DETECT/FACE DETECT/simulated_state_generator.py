# simulated_state_generator.py
import random
import time
from threading import Thread
import numpy as np

# Emotion simulation states
SIMULATED_STATES = ["calm", "stressed", "drowsy", "focused", "happy", "tired"]

# State transition probabilities (more likely to stay in current state or transition to similar states)
TRANSITION_PROBS = {
    "calm": {"calm": 0.6, "focused": 0.2, "happy": 0.1, "stressed": 0.05, "drowsy": 0.03, "tired": 0.02},
    "stressed": {"stressed": 0.5, "calm": 0.2, "focused": 0.15, "tired": 0.1, "drowsy": 0.03, "happy": 0.02},
    "drowsy": {"drowsy": 0.6, "tired": 0.2, "calm": 0.1, "focused": 0.05, "stressed": 0.03, "happy": 0.02},
    "focused": {"focused": 0.5, "calm": 0.2, "stressed": 0.15, "tired": 0.1, "drowsy": 0.03, "happy": 0.02},
    "happy": {"happy": 0.5, "calm": 0.2, "focused": 0.15, "stressed": 0.1, "tired": 0.03, "drowsy": 0.02},
    "tired": {"tired": 0.5, "drowsy": 0.2, "calm": 0.15, "stressed": 0.1, "focused": 0.03, "happy": 0.02}
}

class SimulatedStateGenerator:
    def __init__(self, interval=10):
        self.current_state = "calm"
        self.interval = interval
        self.running = False
        self.state_history = []
        self.max_history = 5

    def start(self):
        self.running = True
        Thread(target=self._run, daemon=True).start()

    def _transition_state(self):
        probs = TRANSITION_PROBS[self.current_state]
        states = list(probs.keys())
        probabilities = list(probs.values())
        return np.random.choice(states, p=probabilities)

    def _run(self):
        while self.running:
            new_state = self._transition_state()
            self.state_history.append(new_state)
            if len(self.state_history) > self.max_history:
                self.state_history.pop(0)
            self.current_state = new_state
            time.sleep(self.interval)

    def stop(self):
        self.running = False

    def get_state(self):
        return self.current_state

    def get_state_history(self):
        return self.state_history.copy()

if __name__ == "__main__":
    sim = SimulatedStateGenerator(interval=5)
    sim.start()
    try:
        while True:
            print("Current State:", sim.get_state())
            print("State History:", sim.get_state_history())
            time.sleep(5)
    except KeyboardInterrupt:
        sim.stop()
        print("Stopped simulation.")
