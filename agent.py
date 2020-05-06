import math
import random
import numpy
import pandas

import gym

import torch.optim
import torch.nn.functional

from network import DeepQNetwork
import constants
import memory
import utils
import test


def compute_epsilon(steps_done):
    if steps_done < 1000000:
        return (-9 / 10000000) * steps_done + 1
    else:
        return 0.1


def select_action(state, policy_nn, steps_done, env):
    epsilon_threshold = compute_epsilon(steps_done)
    sample = random.random()
    if sample > epsilon_threshold:
        with torch.no_grad():
            return policy_nn(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(env.action_space.n)]], dtype=torch.long)


def optimize_model(target_nn, policy_nn, memory, optimizer):
    if len(memory) < constants.BATCH_SIZE:
        return

    transitions = memory.sample(constants.BATCH_SIZE)

    # Array of True/False if the state is not final
    non_final_mask = torch.tensor(tuple(map(lambda t: t.next_state is not None, transitions)))

    non_final_next_states = torch.cat([trans.next_state for trans in transitions
                                       if trans.next_state is not None])
    state_batch = torch.cat([trans.state for trans in transitions])
    action_batch = torch.cat([trans.action for trans in transitions])
    reward_batch = torch.cat([trans.reward for trans in transitions])

    state_action_values = policy_nn(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(constants.BATCH_SIZE)
    next_state_values[non_final_mask] = target_nn(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * constants.GAMMA) + reward_batch
    loss = torch.nn.functional.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def get_screen(env):
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    return utils.transform_image(screen)


def main_training_loop():

    fixed_states = test.get_fixed_states()

    env = gym.make('BreakoutNoFrameskip-v0')

    n_actions = env.action_space.n

    policy_net = DeepQNetwork(constants.STATE_IMG_HEIGHT,
                              constants.STATE_IMG_WIDTH,
                              constants.N_IMAGES_PER_STATE,
                              n_actions)

    target_net = DeepQNetwork(constants.STATE_IMG_HEIGHT,
                              constants.STATE_IMG_WIDTH,
                              constants.N_IMAGES_PER_STATE,
                              n_actions)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.RMSprop(policy_net.parameters())
    replay_memory = memory.ReplayMemory(constants.REPLAY_MEMORY_SIZE)

    steps_done = 0
    epoch = 0
    information = [["epoch", "n_steps", "avg_reward", "avg_score", "n_episodes", "avg_q_value"]]
    try:
        for i_episode in range(constants.N_EPISODES):
            cumulative_screenshot = []

            # Prepare the cumulative screenshot
            padding_image = torch.zeros((1, constants.STATE_IMG_HEIGHT, constants.STATE_IMG_WIDTH))
            for i in range(constants.N_IMAGES_PER_STATE - 1):
                cumulative_screenshot.append(padding_image)

            env.reset()
            episode_score = 0
            episode_reward = 0

            screen_grayscale_state = get_screen(env)
            cumulative_screenshot.append(screen_grayscale_state)

            state = utils.process_state(cumulative_screenshot)

            prev_state_lives = constants.INITIAL_LIVES

            for i in range(constants.N_TIMESTEP_PER_EP):
                if constants.SHOW_SCREEN:
                    env.render()

                action = select_action(state, policy_net, steps_done, env)
                _, reward, done, info = env.step(action)
                episode_score += reward
                episode_reward += reward
                reward_tensor = torch.tensor([reward])

                # Computes current state
                screen_grayscale = get_screen(env)
                cumulative_screenshot.append(screen_grayscale)
                cumulative_screenshot.pop(0)  # Deletes the first element of the list to save memory space

                if done:
                    next_state = None
                else:
                    next_state = utils.process_state(cumulative_screenshot)

                replay_memory.push(state, action, next_state, reward_tensor)

                # Updates state
                state = next_state

                optimize_model(target_net, policy_net, replay_memory, optimizer)
                steps_done += 1

                if done:
                    print("Episode:", i_episode, "Steps done:", steps_done, "- Episode reward:", episode_reward, "- Episode score:", episode_score)
                    break

                # Update target policy
                if steps_done % constants.TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

                # Epoch test
                if steps_done % constants.STEPS_PER_EPOCH == 0:
                    epoch += 1
                    epoch_reward_average, epoch_score_average, n_episodes, q_values_average = test.test_agent(target_net, fixed_states)
                    information.append([epoch, steps_done, epoch_reward_average, epoch_score_average, n_episodes, q_values_average])

        # Save test information in dataframe
        print("Saving information...")
        information_numpy = numpy.array(information)
        dataframe_information = pandas.DataFrame(columns=information_numpy[0, 0:],
                                                 data=information_numpy[1:, 0:])
        dataframe_information.to_csv("info/results.csv")
        print(dataframe_information)

        # Save target parameters in file
        torch.save(target_net.state_dict(), "info/nn_parameters.txt")

    except KeyboardInterrupt:
        # Save test information in dataframe
        print("Saving information...")
        information_numpy = numpy.array(information)
        dataframe_information = pandas.DataFrame(columns=information_numpy[0, 0:], data=information_numpy[1:, 0:])
        dataframe_information.to_csv("info/results.csv")
        print(dataframe_information)

        # Save target parameters in file
        torch.save(target_net.state_dict(), "info/nn_parameters.txt")

    env.close()


if __name__ == "__main__":
    main_training_loop()
