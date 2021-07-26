import sys
sys.path.insert(0,"/Users/jmccalmon/PycharmProjects/qmix")
from runner import Runner
from env.starcraftenv import StarCraft2Env
from arguments import get_common_args, get_mixer_args 
import numpy as np

if __name__ == '__main__':

    rates = [.25, .5, .5, .75, 1]
    thresholds = [2, .2, .2, .03, -1]

    for i in range(3):
        args = get_common_args()
        args = get_mixer_args(args)
        env = StarCraft2Env(map_name=args.map_name,
                            step_mul=args.step_mul,
                            difficulty=args.difficulty,
                            game_version=args.game_version,
                            replay_dir=args.replay_dir)
        env_info = env.get_env_info()
        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.state_shape = env_info["state_shape"]
        #print("No of Action  = ",env_info["n_actions"])
        #print("No of Agents  = ",env_info["n_agents"])
        #print("Obs shape = ",env_info["obs_shape"])
        #print("Episode_limit = ",env_info["episode_limit"])
        args.obs_shape = env_info["obs_shape"]
        args.episode_limit = env_info["episode_limit"]
        args.alg = "qmix"    
        #args.attack_name = "strategic"    #"random"   "random_time"  "strategic"

            #set threshold for strategic time attack
        #args.victim_agent = 0              # victim agent to attack
        #args.attack_rate = 0.25           #Theshold value for the frequency of attack
        #args.adversary = False          #False = Optimal Qmix; True = If you want to enforce any of the attack ("random"   "random_time"  "strategic")

        args.attack_rate = 1
        args.counterfactual_threshold = -1
        args.strategic_threshold = -1

        """
        if i < 5:
            args.attack_rate = rates[i]
            args.counterfactual_threshold = thresholds[i]
        else:
            args.attack_rate = rates[i-5]
            args.counterfactual_threshold = thresholds[i-5]
        """

        if args.evaluate_epoch != 6500:
            save = False
        else:
            print('Will save transition')
            save = True

        if i == 0:
            args.attack_name = "random_time"
        elif i == 1:
            args.attack_name = "strategic"
        else:
            args.attack_name = "counterfactual"

        runner = Runner(env, args)
        if not args.evaluate:
            runner.run(i)
        else:
            win_rate, avg_reward, transitions = runner.evaluate()
            #if args.white_box:
            if i < 5:
                fname = "BB_"
            else:
                fname = "WB_"
            if args.adversary:

                print('Evaluated the {} adversary'.format(args.attack_name))
                if i == 2 or i == 7:
                    fname = fname + "SC_{}_{}_test".format(args.attack_name, args.attack_rate)
                else:
                    fname = fname + "SC_{}_{}".format(args.attack_name, args.attack_rate)

                attack_rate = np.sum(transitions[:, 5]) / np.sum(transitions[:, 7])

                print('Attack rate was {}'.format(attack_rate))
            else:
                print('Evaluated the Optimal Policy')
                fname = fname + "optimal"

            print('The win rate of {} is  {}'.format(args.alg, win_rate))
            print('The average reward is {}'.format(avg_reward))

            print('Transition Shape: {}'.format(transitions[:, :7].shape))

            if save:
                print('Saving Transition...')

                np.save("Transitions/{}".format(fname), transitions[:, :7])







        env.close()


