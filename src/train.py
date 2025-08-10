
    # Run an episode and use experience replay for training
    def run_episode(self, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.7,  batch_size=64, gamma=0.99, max_steps=100, total_episodes=2000):
        state_size = 5  # State can include agent position, target position, whether the agent has the full secret
        action_size = 4
        agent_dqn = DQN(state_size, action_size)

        # Create four agents of two different types sharing the same DQN
        agent1 = Agent(state_size, action_size, "Type1", agent_dqn)
        agent2 = Agent(state_size, action_size, "Type1", agent_dqn)
        agent3 = Agent(state_size, action_size, "Type2", agent_dqn)
        agent4 = Agent(state_size, action_size, "Type2", agent_dqn)

        self.add_agent(agent1)
        self.add_agent(agent2)
        self.add_agent(agent3)
        self.add_agent(agent4)
        
        # Reset success count at the beginning of the training
        self.success_count = 0  # Reset success count before training starts
        
        

        # Training loop
        for episode in range(total_episodes):  # Run multiple training episodes
            self.reset()
            self.met_agents_pairs.clear()  # Clear exchanged agent pairs at the start of each episode
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < max_steps:
                step_count += 1
                
                # Randomize the order of agent actions in each step
                random.shuffle(self.agents)

                actions = []
                # Each agent selects and executes an action
                for agent in self.agents:
                    # Create state (agent position, target position, whether the agent has the full secret)
                    state = np.array([agent.pos[0], agent.pos[1], int(agent.has_full_secret), self.target_location[0], self.target_location[1]])
                    action = agent.select_action(state, epsilon)
                    actions.append(action)
                
                next_states, rewards, done = self.step(actions)
                total_reward += sum(rewards)

                additional_rewards = self.handle_meeting()
                total_reward += sum(additional_rewards)

                for i, agent in enumerate(self.agents):
                    self.replay_buffer.add(state, actions[i], rewards[i], next_states[i], done)
                    
            if done and step_count <= max_steps:
                self.success_count += 1  # Record success count

            # Calculate success rate
            current_success_rate = self.success_count / (episode + 1)
            self.rewards_per_episode.append(total_reward)
            self.success_rate.append(current_success_rate)
            
           

            # Epsilon decay: decay epsilon after each episode
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay  # Decay epsilon
                epsilon = max(epsilon_min, epsilon)  # Ensure it doesn't go below epsilon_min
                        
            # Update the target network every 100 episodes
            if episode % 20 == 0:
                agent_dqn.update_target()  # Sync target network with the prediction network

            print(f"Episode {episode + 1}: {'Success' if done else 'Failed'}. Steps: {step_count}, Total Reward: {total_reward}")

            # Sample from replay buffer and train
            if self.replay_buffer.size() > batch_size:
                states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
                agent_dqn.train_one_step(states, actions, rewards, next_states, dones, gamma)

        # Plot training results
        self.plot_results()
        
        return agent_dqn

    # Plot training results (success rate and total reward over episodes)
    def plot_results(self):
        episodes = np.arange(len(self.rewards_per_episode))

        # Create figure
        plt.figure(figsize=(12, 6))

        # Plot success rate
        plt.subplot(1, 2, 1)
        plt.plot(episodes, self.success_rate, label='Success Rate')
        plt.xlabel('Episodes')
        plt.ylabel('Success Rate')
        plt.title('Success Rate over Episodes')
        plt.legend()

        # Plot total reward
        plt.subplot(1, 2, 2)
        plt.plot(episodes, self.rewards_per_episode, label='Total Reward')
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.title('Total Reward over Episodes')
        plt.legend()

        # Show figure
        plt.tight_layout()
        plt.show()

# Run environment simulation
env = Environment()
agent_dqn = env.run_episode(max_steps=100, total_episodes=20000)
