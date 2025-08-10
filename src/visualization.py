import matplotlib.pyplot as plt
import numpy as np

def visualize_test_results(test_results):
    """
    Visualize the test results from the test_agent function.
    
    Args:
        test_results: A dictionary containing the test results with keys:
                      - 'success_rate': overall success rate
                      - 'avg_steps': average steps per episode
                      - 'avg_reward': average reward per episode
                      - 'total_steps': list of total steps per episode
                      - 'total_rewards': list of total rewards per episode
    """
    
    episodes = np.arange(len(test_results['total_rewards']))
    
    # Create figure for the three plots
    plt.figure(figsize=(16, 8))
    
    # Plot 1: Total Rewards per Episode
    plt.subplot(1, 3, 1)
    plt.plot(episodes, test_results['total_rewards'], label='Total Rewards', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Total Rewards')
    plt.title('Total Rewards per Episode')
    plt.legend()

    # Plot 2: Total Steps per Episode
    plt.subplot(1, 3, 2)
    plt.plot(episodes, test_results['total_steps'], label='Total Steps', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Total Steps')
    plt.title('Total Steps per Episode')
    plt.legend()

    # Plot 3: Success Rate (Single Value)
    plt.subplot(1, 3, 3)
    plt.bar(['Success Rate'], [test_results['success_rate'] * 100], color='orange')
    plt.ylim([0, 100])
    plt.ylabel('Success Rate (%)')
    plt.title('Overall Success Rate')

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

# Example of visualizing the test results
visualize_test_results(test_results)
