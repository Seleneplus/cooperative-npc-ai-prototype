import random
from collections import deque

class ReplayBuffer:
    """
    Simple replay buffer for storing and sampling experiences.
    Each experience: (state, action, reward, next_state, done)
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """Store a single experience tuple."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Randomly sample a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def size(self):
        """Return the current size of the buffer."""
        return len(self.buffer)
