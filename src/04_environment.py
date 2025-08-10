# Define the Environment class, managing the environment where agents operate
class Environment:
    def __init__(self):
        self.grid_size = 5  # Size of the environment grid
        self.target_location = [random.randint(0, 4), random.randint(0, 4)]  # Random target location
        self.agents = []  # List to store agents
        self.step_penalty = -0.5  # Small negative reward for each step
        self.meet_reward = 50  # Reward for exchanging information between agents
        self.goal_reward = 200  # Reward for reaching the target
        self.met_agents_pairs = set()  # Set to store pairs of agents that have exchanged information
        self.replay_buffer = ReplayBuffer(capacity=100000)  # Replay buffer for experience replay

        # Records for tracking the training progress
        self.rewards_per_episode = []
        self.success_rate = []
        self.success_count = 0

    # Add an agent to the environment
    def add_agent(self, agent):
        self.agents.append(agent)

    # Reset the environment at the beginning of an episode
    def reset(self):
        self.target_location = [random.randint(0, 4), random.randint(0, 4)]  # Reset the target location
        for agent in self.agents:
            agent.pos = [random.randint(0, 4), random.randint(0, 4)]  # Reset agent positions
            agent.has_full_secret = False
            agent.met_agent = False
        state = np.array([self.agents[0].pos[0], self.agents[0].pos[1], int(self.agents[0].has_full_secret), self.target_location[0], self.target_location[1]])
        return state

    # Check if any agent has reached the target location
    def check_goal(self):
        for agent in self.agents:
            if agent.pos == self.target_location and agent.has_full_secret:
                return True
        return False
    
    # Perform one step in the environment for all agents
    def step(self, actions):
        next_states = []
        done = False
        
        # Each agent takes its action and updates its position
        for i, agent in enumerate(self.agents):
            action = actions[i]
            agent.update_position(action)  # Update each agent's position based on their action

        # Create the next state for each agent
        for agent in self.agents:
            next_state = np.array([agent.pos[0], agent.pos[1], int(agent.has_full_secret), self.target_location[0], self.target_location[1]])
            next_states.append(next_state)

        # Check if any agent has reached the goal
        if any(agent.pos == self.target_location and agent.has_full_secret for agent in self.agents):
            done = True

        # Calculate the reward for all agents
        reward = self.calculate_rewards()

        return next_states, reward, done  # Return next state, reward, and done flag

    # Visualize the grid showing the positions of agents and the target
    def plot_grid(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_xticks(range(self.grid_size + 1))
        ax.set_yticks(range(self.grid_size + 1))
        ax.grid(True)

        # Draw the target location
        target_circle = patches.Circle((self.target_location[0], self.target_location[1]), 0.4, color='green', label="Target")
        ax.add_patch(target_circle)
        plt.text(self.target_location[0] - 0.2, self.target_location[1], 'B', fontsize=12, color='white', weight='bold')

        # Draw the agents' positions
        colors = ['blue', 'red', 'yellow', 'black']
        for i, agent in enumerate(self.agents):
            agent_circle = patches.Circle((agent.pos[0], agent.pos[1]), 0.2, color=colors[i], label=f"Agent {i+1}")
            ax.add_patch(agent_circle)

        plt.legend()
        plt.show()

    # Calculate the rewards for the agents
    def calculate_rewards(self):
        rewards = [self.step_penalty] * len(self.agents)
        for i, agent in enumerate(self.agents):
            if agent.has_full_secret and agent.pos == self.target_location:
                rewards[i] = self.goal_reward  # If the agent reached the goal, give full reward
            elif agent.has_full_secret:
                # Reward for getting closer to the target
                distance_to_goal = abs(agent.pos[0] - self.target_location[0]) + abs(agent.pos[1] - self.target_location[1])
                rewards[i] += (5 - distance_to_goal)  # Reward inversely proportional to 
               
            
        return rewards

    # Handle meetings between agents and return additional rewards
    def handle_meeting(self):
        additional_rewards = [0] * len(self.agents)
        
        # Create pairs of agents for potential meetings (only different types should exchange information)
        agent_pairs = [(0, 2), (0, 3), (1, 2), (1, 3)]

        for pair in agent_pairs:
            agent1, agent2 = self.agents[pair[0]], self.agents[pair[1]]
            
            # Check if they are of different types and meet
            if agent1.agent_type != agent2.agent_type and agent1.meet(agent2):
                # Check if this pair has not already exchanged information
                if (pair[0], pair[1]) not in self.met_agents_pairs:
                    self.met_agents_pairs.add((pair[0], pair[1]))  # Record that these agents met
                    additional_rewards[pair[0]] += self.meet_reward  # Reward both agents for exchanging information
                    additional_rewards[pair[1]] += self.meet_reward
                    
        return additional_rewards

   
