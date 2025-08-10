# Define the DQN class, representing the Q-network and its target network
class DQN:
    def __init__(self, state_size=5, action_size=4):
        # Define the architecture of the Q-network
        l1 = state_size
        l2 = 24
        l3 = 24
        l4 = action_size

        # Create the Q-network as a sequence of layers
        self.model = nn.Sequential(
            nn.Linear(l1, l2),
            nn.ReLU(),
            nn.Linear(l2, l3),
            nn.ReLU(),
            nn.Linear(l3, l4)
        )

        # Create a target network by copying the Q-network
        self.target_model = copy.deepcopy(self.model)
        self.loss_fn = nn.MSELoss()  # Mean squared error loss function
        self.learning_rate = 0.001
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)  # Use Adam optimizer

    # Update the target network to match the Q-network's weights
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # Predict Q-values for a given state using the Q-network
    def get_qvals(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)  # Convert state to tensor
        q_values = self.model(state)  # Forward pass through the Q-network
        return q_values

    # Get the maximum Q-value from the target network for a given state
    def get_maxQ(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)  # Convert state to tensor
        q_values = self.target_model(state)  # Forward pass through the target network
        return torch.max(q_values).item()  # Return the maximum Q-value

    # Train the Q-network using a batch of experiences
    def train_one_step(self, states, actions, rewards, next_states, dones, gamma=0.99):
        # Convert states and next_states to tensors
        states_batch = torch.cat([torch.from_numpy(np.array(s)).float().unsqueeze(0) for s in states])
        next_states_batch = torch.cat([torch.from_numpy(np.array(s)).float().unsqueeze(0) for s in next_states])

        # Convert actions, rewards, and dones to tensors
        action_batch = torch.tensor(actions).long()
        reward_batch = torch.tensor(rewards).float()
        done_batch = torch.tensor(dones).float()

        # Get current Q-values for the actions taken
        Q1 = self.model(states_batch)
        current_Q = Q1.gather(1, action_batch.unsqueeze(1)).squeeze()

        # Get the maximum Q-values for the next states from the target network
        next_Q = self.target_model(next_states_batch).max(1)[0]
        target_Q = reward_batch + (gamma * next_Q * (1 - done_batch))  # Compute target Q-values

        # Compute the loss between current Q-values and target Q-values
        loss = self.loss_fn(current_Q, target_Q.detach())

        # Perform backpropagation and update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
