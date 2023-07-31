import environments
import agents

env = environments.GridWorld(grid_size=4, hole_count=4)
agent = agents.MonteCarloAgent(
    num_states=env.state_space.shape[0],
    num_actions=env.action_space.shape[0])

# Train the agent
agent.first_visit_mc_prediction(100, env)

# Now the agent's policy should be improved
print("Final policy:", agent.policy)
