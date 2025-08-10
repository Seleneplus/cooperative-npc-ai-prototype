def test_agent(agent_dqn, num_episodes=100, max_steps=15):
    """
    Test the trained DQN agent over a number of episodes.
    
    Args:
        agent_dqn: Trained DQN agent to be tested.
        num_episodes: The number of episodes to run the test.
        max_steps: The maximum number of steps allowed for success (default is 15).
        
    Returns:
        dict: A dictionary containing success rate, average reward, and average steps per episode.
    """
    # Create a new environment for testing
    env = Environment()
    
    success_count = 0  # Track the number of successful episodes
    total_steps = []   # List to store the number of steps in each episode
    total_rewards = [] # List to store the total rewards in each episode
    
    # Create 4 agents using the trained DQN, 2 agents of type "Type1" and 2 of "Type2"
    agent1 = Agent(state_size=3, action_size=4, agent_type="Type1", dqn=agent_dqn)
    agent2 = Agent(state_size=3, action_size=4, agent_type="Type1", dqn=agent_dqn)
    agent3 = Agent(state_size=3, action_size=4, agent_type="Type2", dqn=agent_dqn)
    agent4 = Agent(state_size=3, action_size=4, agent_type="Type2", dqn=agent_dqn)
    
    # Add the agents to the environment
    env.add_agent(agent1)
    env.add_agent(agent2)
    env.add_agent(agent3)
    env.add_agent(agent4)
    
    # Dictionary to store each agent's movement paths during the episodes
    agent_paths = {i: [] for i in range(len(env.agents))}
    
    # Initialize path_lengths to store the length of each agent's path
    path_lengths = {i: [] for i in range(len(env.agents))}

    # Loop through each test episode
    for episode in range(num_episodes):
        env.reset()  # Reset the environment for the new episode
        done = False  # Flag to track if the episode has ended
        episode_reward = 0  # Track total rewards for this episode
        step_count = 0  # Track the number of steps taken
        
        # Continue the episode until done or max_steps is reached
        while not done and step_count < max_steps:
            step_count += 1  # Increment step count
            
            # Each agent takes an action based on the current state
            for idx, agent in enumerate(env.agents):
                # Define the state (agent position, full secret status, target location)
                state = np.array([agent.pos[0], agent.pos[1], int(agent.has_full_secret), env.target_location[0], env.target_location[1]])
                
                # Select action based on the current policy (epsilon = 0, meaning no exploration)
                action = agent.select_action(state, epsilon=0)
                agent.update_position(action)  # Update the agent's position based on the action
                
                # Record the agent's position after taking the action
                agent_paths[idx].append((agent.pos[0], agent.pos[1]))
            
            # Handle agent meetings and calculate rewards
            additional_rewards = env.handle_meeting()  # Agents can meet and exchange information
            rewards = env.calculate_rewards()  # Calculate rewards based on their positions and goals
            episode_reward += sum(rewards) + sum(additional_rewards)  # Add rewards to the total reward
            
            # After exchanging secrets, agents with the full secret continue towards the goal
            for agent in env.agents:
                if agent.has_full_secret:
                    state = np.array([agent.pos[0], agent.pos[1], int(agent.has_full_secret), env.target_location[0], env.target_location[1]])
                    action = agent.select_action(state, epsilon=0)  # Continue selecting actions based on policy
                    agent.update_position(action)  # Update the agent's position
                    agent_paths[idx].append((agent.pos[0], agent.pos[1]))  # Record new position after action
            
            # Check if any agent has reached the goal location
            if env.check_goal():
                done = True  # If any agent reaches the goal, mark the episode as done
        
        # Track the results of the current episode
        total_steps.append(step_count)  # Record the number of steps taken
        total_rewards.append(episode_reward)  # Record the total reward for the episode
        
        
        for idx, path in agent_paths.items():
            path_lengths[idx].append(len(path))  # Record the length of the path (number of steps)
            
            
        # If the episode ends successfully within the maximum steps, count it as a success
        if done and step_count <= max_steps:
            success_count += 1  # Increment the success count
        
        # Print the results of the current episode
        print(f"Episode {episode + 1}: Steps = {step_count}, Reward = {episode_reward}, Success = {'Yes' if done and step_count <= max_steps else 'No'}")
    
    # Calculate the overall success rate, average number of steps, and average rewards across all episodes
    success_rate = success_count / num_episodes  # Success rate across all episodes
    avg_steps = np.mean(total_steps)  # Average number of steps per episode
    avg_reward = np.mean(total_rewards)  # Average reward per episode
    
    # Print the final test results
    print(f"\n=== Test Results ===")
    print(f"Success Rate: {success_rate * 100:.2f}%")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Average Reward: {avg_reward:.2f}")
    
    # Return the results in a dictionary
    return {
        'success_rate': success_rate,
        'avg_steps': avg_steps,
        'avg_reward': avg_reward,
        'total_steps': total_steps,
        'total_rewards': total_rewards,
        'path_lengths': path_lengths,  # Add path lengths to the results
        'agent_paths': agent_paths  # Include the paths of all agents for visualization
    }

# Example usage: test the trained agent with 100 episodes and max 15 steps per episode
test_results = test_agent(agent_dqn=agent_dqn, num_episodes=100, max_steps=15)
