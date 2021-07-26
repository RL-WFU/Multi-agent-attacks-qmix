import numpy as np
import torch
from torch.distributions import one_hot_categorical
import time
from numpy import save
import sys
sys.path.insert(0,"/root/qmix")


class RolloutWorker:
    def __init__(self, env, agents, args, adversarial):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args
        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        self.adversarial = adversarial
        
        print('Successful Init of RolloutWorker')
    #===============>entry 1

    def generate_transition(self, obs, actions, next_obs, attacked, valid_step, able_to_attack):

        obs = np.asarray(obs).flatten()
        next_obs = np.asarray(next_obs).flatten()
        actions = np.asarray(actions).flatten()
        #print(obs.shape)
        #print(actions.shape)
        #print(next_obs.shape)
        transition = (obs, actions[0], actions[1], actions[2], next_obs, attacked, valid_step, able_to_attack)
        return transition

    def generate_episode(self, episode_num=None, evaluate=False, black_box=False, bb_net=None):

        if black_box:
            assert bb_net is not None, "Specify black box network"


        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay of evaluation
            self.env.close()
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        a, adv_o, adv_action, threshold = [], [], [], []
        cor_obs = []
        cor_acts = []

        all_obs = []
        
        self.env.reset()
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0  # cumulative rewards
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        #==================================================
        #==============HIDDEN INITIALIZED =================
        self.agents.policy.init_hidden(1)

        transitions = []

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        while not terminated and step < self.episode_limit:
            # time.sleep(0.2)
            obs = self.env.get_obs() #(3, 30) - agent observations
            state = self.env.get_state() #(48,) Global state - should not be passed during execution

            all_obs.append(np.reshape(np.asarray(obs), [1, 90]))

            attacked = 0
            able_to_attack = 1

            actions, avail_actions, actions_onehot = [], [], []
            for agent_id in range(self.n_agents):
                #print("Agent ID=",  agent_id)
                avail_action = self.env.get_avail_agent_actions(agent_id)
                #===========================================================
                #=========IF STATE PERTURBATION CHANGE HERE ================
                #===========================================================

                if self.args.victim_agent == agent_id:
                    cor_obs.append(obs[agent_id])
                if self.args.victim_agent == agent_id and self.args.adversary:

                  #USE ADVERSARIAL ATTACK HERE
                  #print("Attack launched")
                  if self.args.attack_name == "random" and np.random.uniform() <= self.args.attack_rate:
                    if black_box:
                        pass
                    else:
                        action = self.adversarial.random_attack(obs[agent_id], last_action[agent_id], agent_id,
                                                       avail_action, epsilon, evaluate)
                    attacked = 1
                  elif self.args.attack_name == "random_time"  and np.random.uniform() <= self.args.attack_rate:
                    #self.adversarial.policy.init_hidden(1)
                    if black_box: #TODO: include last action in prediction net
                        if step < 2:
                            action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                               avail_action, epsilon, evaluate)
                            attacked = 0
                            able_to_attack = 0
                        else:

                            input_obs = np.asarray(cor_obs[-3:])
                            input_obs = np.expand_dims(input_obs, axis=0)
                            probs = bb_net.predict_on_batch(input_obs)
                            avail = torch.tensor(avail_action, dtype=torch.float32).unsqueeze(0)
                            probs[avail == 0.0] = float("inf")
                            action = np.argmin(probs)
                            able_to_attack = 1
                            attacked = 1
                    else:
                        q_val = self.agents.get_qvalue(obs[agent_id], last_action[agent_id], agent_id, avail_action, epsilon, evaluate)
                        action = self.adversarial.random_time_attack(q_val, avail_action)
                        attacked = 1
                    #print("attack successful")
                  elif self.args.attack_name == "strategic":
                    if black_box:
                        if step < 2:
                            action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                               avail_action, epsilon, evaluate)
                            attacked = 0
                            able_to_attack = 0
                        else:

                            input_obs = np.asarray(cor_obs[-3:])
                            input_obs = np.expand_dims(input_obs, axis=0)
                            probs = bb_net.predict_on_batch(input_obs)
                            avail = torch.tensor(avail_action, dtype=torch.float32).unsqueeze(0)
                            probs[avail == 0.0] = -1 * float("inf")
                            maximum = np.max(probs)
                            probs[avail == 0.0] = float("inf")
                            minimum = np.min(probs)
                            c = maximum - minimum
                            if c > self.args.strategic_threshold:
                                action = np.argmin(probs)
                                attacked = 1
                            else:
                                action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                       avail_action, epsilon, evaluate)
                                attacked = 0

                            able_to_attack = 1

                    else:
                        demo_thrs = self.args.strategic_threshold
                        q_val = self.agents.get_qvalue(obs[agent_id], last_action[agent_id], agent_id, avail_action,
                                                       epsilon, evaluate)
                        action, diff, attacked = self.adversarial.strategic_time_attack(q_val, avail_action, epsilon,
                                                                                        demo_thrs)
                        threshold.append(diff)
                        if action == 0:
                            able_to_attack = 0
                    # action = self.agents.choose_strategic_action(obs[agent_id], last_action[agent_id], agent_id,
                    #                                    avail_action, epsilon, evaluate)
                    # action = self.adversarial_policy(obs[agent_id],avail_action)

                  elif self.args.attack_name == 'counterfactual':
                    if black_box:

                        #bb_net[0]: corrupted agent prediction
                        #shape: [1, 3, 30]
                        #bb_net[1]: other agents prediction
                        #shape: [1, 3, 31]
                        #bb_net[2]: transition prediction
                        #shape: [1, 3, 33]
                        """
                        At time t:
                        1. Predict agent 0 action given obs from [t-2, t-1, t]
                            Input shape: [1, 3, 30] Output shape: [1, 9]. Take argmax (after 0ing unavailable)
                        2. Predict agent 1 and 2 action given obs and agent 0 action from [t-2, t-1, t]
                            Input shape: [1, 3, 91] Output shape: [1, 18]?. Take argmax (after 0ing unavailable)
                        3. Predict next state given obs, actions of agents 1 and 2, and all possible actions of 0, from [t-2, t-1, t]
                            Input shape: [9, 3, 93] Output shape: [9, 90]
                        4. Find optimal agent 0 action from states [t-1, t, t+1], where obs t+1 is predicted
                            Input shape: [9, 3, 30] (note: [i, :2, 30] == [j, :2, 30] for all i, j) Output shape: [9, 9] 
                        5. Find agent 1 and 2 actions for states and agent 0 actions [t-1, t, t+1], for every predicted state
                            Input shape: [9, 3, 91] Output shape: [9, 2, 9] (denote attack_probs)
                        6. Calculate kl divergence between attack_probs[optimal] and attack_probs / attack_probs[optimal]
                        7. Attacking action = np.argmax(kl divergence)
                        """
                        assert isinstance(bb_net, list), "Need multiple black box networks"




                        if step < 2:
                            action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                               avail_action, epsilon, evaluate)
                            attacked = 0
                            able_to_attack = 0
                        else:
                            past_actions = np.asarray(a[-2:])
                            past_cor_actions = past_actions[:, 0]
                            past_other_actions = past_actions[:, 1:]
                            assert past_other_actions.shape == (2, 2), "Should be (2, 2), was {}".format(past_other_actions.shape)
                            #PREDICTION NETWORKS FOR OTHER AGENTS - NOT JUST TEAMMATES
                            cor_input_obs = np.asarray(cor_obs[-3:])
                            cor_input_obs = np.expand_dims(cor_input_obs, axis=0)

                            all_input_obs = np.asarray(all_obs[-3:])
                            all_input_obs = np.reshape(np.expand_dims(all_input_obs, axis=0), [1, 3, 90])

                            assert all_input_obs.shape == (1, 3, 90), "Should be (1, 3, 90), was {}".format(all_input_obs.shape)
                            assert cor_input_obs.shape == (1, 3, 30), "Should be (1, 3, 30), was {}".format(cor_input_obs.shape)

                            probs = bb_net[0].predict_on_batch(cor_input_obs)

                            assert probs.shape == (1, 9), "Should be (1, 9), was {}".format(probs.shape)

                            avail = torch.tensor(avail_action, dtype=torch.float32).unsqueeze(0)
                            probs[avail == 0.0] = -1 * float("inf")
                            corrupt_action = np.argmax(probs)

                            corrupt_action = np.expand_dims(np.asarray(corrupt_action), axis=0)
                            past_cor_actions = np.expand_dims(np.concatenate([past_cor_actions, corrupt_action], axis=0), axis=1)
                            past_cor_actions = np.expand_dims(past_cor_actions, axis=0)
                            assert past_cor_actions.shape == (1, 3, 1), "Should be (1, 3, 1), was {}".format(past_cor_actions.shape)


                            temp_inputs = np.concatenate([all_input_obs, past_cor_actions], axis=2)
                            assert temp_inputs.shape == (1, 3, 91), "Should be (1, 3, 91), was {}".format(
                                temp_inputs.shape)

                            temp_inputs = np.asarray(temp_inputs, np.float32)
                            o_probs = bb_net[1].predict_on_batch(temp_inputs)
                            o_probs = np.asarray(o_probs)
                            correct_other_probs = np.reshape(o_probs, [1, 18])
                            assert correct_other_probs.shape == (1, 18), "Should be (1, 18), was {}".format(correct_other_probs.shape)

                            correct_actions = np.asarray([np.argmax(np.reshape(correct_other_probs, [2, 9])[0, :]), np.argmax(np.reshape(correct_other_probs, [2, 9])[1, :])])
                            assert correct_actions.shape == (2,), "Should be (2,), was {}".format(correct_actions.shape)
                            correct_actions = np.reshape(correct_actions, [1, 1, 2])
                            past_other_actions = past_other_actions[-2:]
                            past_other_actions = np.reshape(past_other_actions, [1, 2, 2])
                            past_other_actions = np.concatenate([past_other_actions, correct_actions], axis=1)
                            assert past_other_actions.shape == (1, 3, 2), "Should be (1, 3, 2), was {}".format(past_other_actions.shape)

                            temp_inputs = np.concatenate([temp_inputs, past_other_actions], axis=2)
                            assert temp_inputs.shape == (1, 3, 93), "Should be (1, 3, 93), was {}".format(temp_inputs.shape)

                            temp_inputs = np.asarray(temp_inputs, np.float32)
                            correct_next_state = np.expand_dims(bb_net[2].predict_on_batch(temp_inputs), axis=0)
                            assert correct_next_state.shape == (1, 1, 90), "Should be (1, 1, 90), was {}".format(correct_next_state.shape)

                            prev_trans = np.copy(temp_inputs)
                            next_states = []
                            probs = np.squeeze(probs)
                            for i in range(self.n_actions):
                                if probs[i] > -1:
                                    temp_inputs[0, 2, 90] = i
                                    next_states.append(bb_net[2].predict_on_batch(temp_inputs))
                                else:
                                    next_states.append(np.zeros([1, 90]))

                            next_states = np.reshape(np.asarray(next_states), [self.n_actions, 90])
                            assert next_states.shape == (9, 90), "Should be (9, 90), was {}".format(next_states.shape)

                            next_states = np.expand_dims(next_states, axis=1)
                            temp_inputs = [prev_trans[:, -2:, :90] for i in range(9)]
                            temp_inputs = np.asarray(temp_inputs)
                            temp_inputs = np.reshape(temp_inputs, [9, 2, 90])
                            next_states = np.concatenate([temp_inputs, next_states], axis=1)
                            assert next_states.shape == (9, 3, 90), "Should be (9, 3, 90), was {}".format(next_states.shape)
                            next_0_action_probs = bb_net[0].predict_on_batch(next_states[:, :, :30])
                            next_0_action_probs = np.reshape(next_0_action_probs, [9, 9])
                            next_0_actions = []
                            for i in range(self.n_actions):
                                if probs[i] > -1:
                                    next_0_actions.append(np.argmax(next_0_action_probs[i]))
                                else:
                                    next_0_actions.append(-1)

                            next_0_actions = np.reshape(np.asarray(next_0_actions), [9, 1]) #These are actions for time t+1
                            t_m_1_actions = [past_cor_actions[:, -2, :] for i in range(9)]
                            t_m_1_actions = np.reshape(np.asarray(t_m_1_actions), [9, 1]) #Actions for time t-1
                            t_actions = np.reshape(np.arange(9), [9, 1]) #Actions for time t
                            next_0_actions = np.concatenate([np.concatenate([t_m_1_actions, t_actions], axis=1), next_0_actions], axis=1)
                            next_0_actions = np.expand_dims(next_0_actions, axis=2)

                            assert next_0_actions.shape == (9, 3, 1), "Should be (9, 3, 1), was {}".format(next_0_actions.shape)

                            temp_inputs = np.concatenate([next_states[:,:,:90], next_0_actions], axis=2)
                            assert temp_inputs.shape == (9, 3, 91), "Should be (9, 3, 31), was {}".format(temp_inputs)
                            temp_inputs = np.asarray(temp_inputs, np.float32)
                            next_other_probs = []
                            valid_indices = []
                            for i in range(self.n_actions):
                                if probs[i] > -1:
                                    act = bb_net[1].predict_on_batch(np.expand_dims(temp_inputs[i], axis=0))
                                    act = np.reshape(np.asarray(act), [1, 18])
                                    next_other_probs.append(act)

                                    valid_indices.append(1)
                                else:
                                    next_other_probs.append(np.zeros([1, 18]))
                                    valid_indices.append(0)



                            next_other_probs = np.reshape(next_other_probs, [9, 18])
                            kl_divergences = []
                            for i in range(self.n_actions):
                                total_kl = 0
                                if valid_indices[i] == 1:
                                    for j in range(18):
                                        log_term = next_other_probs[corrupt_action, j] / next_other_probs[i, j]
                                        total_kl += next_other_probs[corrupt_action, j] * np.log(log_term)
                                else:
                                    total_kl = -1
                                kl_divergences.append(total_kl)

                            worst_action = np.argmax(kl_divergences)

                            if kl_divergences[worst_action] > self.args.counterfactual_threshold:
                                action = worst_action
                                attacked = 1
                            else:
                                action = corrupt_action
                                attacked = 0

                    else:
                        pass


                  else:
                    action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                       avail_action, epsilon, evaluate)
                else:
                    if self.args.white_box:
                        action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                       avail_action, epsilon, evaluate)
                    else:

                        if self.args.victim_agent == agent_id:
                            if step < 2:
                                action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                                   avail_action, epsilon, evaluate)
                            else:
                                cor_obs.append(obs[agent_id])
                                input_obs = np.asarray(cor_obs[-3:])
                                input_obs = np.expand_dims(input_obs, axis=0)
                                probs = bb_net.predict_on_batch(input_obs)
                                avail = torch.tensor(avail_action, dtype=torch.float32).unsqueeze(0)
                                probs[avail == 0.0] = -1 * float("inf")
                                action = np.argmax(probs)
                        else:
                            action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                               avail_action, epsilon, evaluate)


                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(action)

                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot
                if agent_id == self.args.victim_agent:
                  adv_o.append(obs[agent_id])
                  adv_action.append(action)
            reward, terminated, info = self.env.step(actions)
            if action == 0:
                able_to_attack = 0
                attacked = 0

            win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False
            o.append(obs)
            a.append(actions)



            if step < 25:
                valid_step = 1
                next_obs = self.env.get_obs()
                transition = self.generate_transition(obs, actions, next_obs, attacked, valid_step, able_to_attack)
                transitions.append(transition)

            if terminated and step < 24:
                valid_step = 0
                able_to_attack = 0
                zero_obs = np.zeros((90,))
                zero_acts = np.zeros((3,))
                attacked = 0
                for i in range(24 - step):
                    transition = self.generate_transition(zero_obs, zero_acts, zero_obs, attacked, valid_step, able_to_attack)
                    transitions.append(transition)

            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        # last obs
        obs = self.env.get_obs()
        state = self.env.get_state()
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])
            #a.append(np.zeros((self.n_agents, self.n_actions)))

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
    
        #===========EPISODE SAVED FOR CAM=======================
        data_set = dict(
                        state = s.copy(),
                        observation = o.copy(),
                        action = a.copy(),
                        next_state = s_next.copy(),
                        observation_next=o_next.copy(),
                        )
      
        adv_data = dict(
                        adv_observation = adv_o.copy(),
                        adv_action = adv_action.copy()
                      )
        
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon
   
        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()

        transitions = np.asarray(transitions)
        #print(transitions.shape)
        return episode, episode_reward, win_tag, step, data_set, adv_data, threshold, transitions

