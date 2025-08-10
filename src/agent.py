# Define the Agent class, representing each agent in the environment
class Agent:
    def __init__(self, state_size, action_size, agent_type, dqn):
        self.state_size = state_size  # State size for the agent
        self.action_size = action_size  # Action size for the agent
        self.dqn = dqn  # DQN shared between agents of the same type
        self.agent_type = agent_type  # Type of agent ("Type1" or "Type2")
        self.pos = [random.randint(0, 4), random.randint(0, 4)]  # Random initial position in a 5x5 grid
        self.has_full_secret = False  # Initial state of the agent (no full secret knowledge)
        self.met_agent = False  # Whether the agent has met another agent of a different type
        

    # Select an action using an epsilon-greedy policy
    def select_action(self, state, epsilon=0.1):
        if self.has_full_secret:
            # Move directly towards the target if the agent has the full secret
            target_x, target_y = state[3], state[4]
            agent_x, agent_y = self.pos[0], self.pos[1]
            if agent_x < target_x:
                return 3  # Move east
            elif agent_x > target_x:
                return 2  # Move west
            elif agent_y < target_y:
                return 1  # Move south
            elif agent_y > target_y:
                return 0  # Move north
     # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            return random.randint(0, self.action_size - 1)  # Random action (exploration)
        else:
            q_values = self.dqn.get_qvals(state)  # Get Q-values from the Q-network
            return torch.argmax(q_values).item()  # Action with the highest Q-value (exploitation)

    # Update the agent's position based on the selected action
    def update_position(self, action):
        if action == 0:  # Move north
            self.pos[1] = max(0, self.pos[1] - 1)
        elif action == 1:  # Move south
            self.pos[1] = min(4, self.pos[1] + 1)
        elif action == 2:  # Move west
            self.pos[0] = max(0, self.pos[0] - 1)
        elif action == 3:  # Move east
            self.pos[0] = min(4, self.pos[0] + 1)

    # Check if the agent meets another agent and exchanges secrets
    def meet(self, other_agent):
        if self.pos == other_agent.pos and not self.has_full_secret:
            self.has_full_secret = True  # Exchange secrets with the other agent
            other_agent.has_full_secret = True
            return True  # Meeting was successful
        return False
