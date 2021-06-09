from rlpyt.experiments.configs.atari.dqn.atari_dqn import configs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class dummy_context_mgr:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False

def gen_args():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game', help='Atari game', default='ms_pacman')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--grayscale', type=int, default=1)
    parser.add_argument('--framestack', type=int, default=4)
    parser.add_argument('--imagesize', type=int, default=84)
    parser.add_argument('--n-steps', type=int, default=100000)
    parser.add_argument('--dqn-hidden-size', type=int, default=256)
    parser.add_argument('--target-update-interval', type=int, default=1)
    parser.add_argument('--target-update-tau', type=float, default=1.)
    parser.add_argument('--momentum-tau', type=float, default=0.01)
    parser.add_argument('--batch-b', type=int, default=1)
    parser.add_argument('--batch-t', type=int, default=1)
    parser.add_argument('--beluga', action="store_true")
    parser.add_argument('--jumps', type=int, default=5)
    parser.add_argument('--num-logs', type=int, default=10)
    parser.add_argument('--renormalize', type=int, default=1)
    parser.add_argument('--dueling', type=int, default=1)
    parser.add_argument('--replay-ratio', type=int, default=64)
    parser.add_argument('--dynamics-blocks', type=int, default=0)
    parser.add_argument('--residual-tm', type=int, default=0.)
    parser.add_argument('--n-step', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--tag', type=str, default='', help='Tag for wandb run.')
    parser.add_argument('--wandb-dir', type=str, default='', help='Directory for wandb files.')
    parser.add_argument('--norm-type', type=str, default='bn', choices=["bn", "ln", "in", "none"], help='Normalization')
    parser.add_argument('--aug-prob', type=float, default=1., help='Probability to apply augmentation')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout probability in convnet.')
    parser.add_argument('--spr', type=int, default=1)
    parser.add_argument('--distributional', type=int, default=1)
    parser.add_argument('--delta-clip', type=float, default=1., help="Huber Delta")
    parser.add_argument('--prioritized-replay', type=int, default=1)
    parser.add_argument('--momentum-encoder', type=int, default=1)
    parser.add_argument('--shared-encoder', type=int, default=0)
    parser.add_argument('--local-spr', type=int, default=0)
    parser.add_argument('--global-spr', type=int, default=1)
    parser.add_argument('--noisy-nets', type=int, default=1)
    parser.add_argument('--noisy-nets-std', type=float, default=0.1)
    parser.add_argument('--classifier', type=str, default='q_l1', choices=["mlp", "bilinear", "q_l1", "q_l2", "none"], help='Style of NCE classifier')
    parser.add_argument('--final-classifier', type=str, default='linear', choices=["mlp", "linear", "none"], help='Style of NCE classifier')
    parser.add_argument('--augmentation', type=str, default=["shift", "intensity"], nargs="+",
                        choices=["none", "rrc", "affine", "crop", "blur", "shift", "intensity"],
                        help='Style of augmentation')
    parser.add_argument('--q-l1-type', type=str, default=["value", "advantage"], nargs="+",
                        choices=["noisy", "value", "advantage", "relu"],
                        help='Style of q_l1 projection')
    parser.add_argument('--target-augmentation', type=int, default=1, help='Use augmentation on inputs to target networks')
    parser.add_argument('--eval-augmentation', type=int, default=0, help='Use augmentation on inputs at evaluation time')
    parser.add_argument('--reward-loss-weight', type=float, default=0.)
    parser.add_argument('--model-rl-weight', type=float, default=0.)
    parser.add_argument('--model-spr-weight', type=float, default=5.)
    parser.add_argument('--t0-spr-loss-weight', type=float, default=0.)
    parser.add_argument('--eps-steps', type=int, default=2001)
    parser.add_argument('--min-steps-learn', type=int, default=2000)
    parser.add_argument('--eps-init', type=float, default=1.)
    parser.add_argument('--eps-final', type=float, default=0.)
    parser.add_argument('--final-eval-only', type=int, default=1)
    parser.add_argument('--time-offset', type=int, default=0)
    parser.add_argument('--project', type=str, default="mpr")
    parser.add_argument('--entity', type=str, default="abs-world-models")
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=0)
    parser.add_argument('--max-grad-norm', type=float, default=10., help='Max Grad Norm')
    parser.add_argument('--public', action='store_true', help='If set, uses anonymous wandb logging')
    args = parser.parse_args()
    return args

def set_config(args, game):
    # TODO: Use Hydra to manage configs
    config = configs['ernbw']
    config['env']['game'] = game
    config["env"]["grayscale"] = args.grayscale
    config["env"]["num_img_obs"] = args.framestack
    config["eval_env"]["game"] = config["env"]["game"]
    config["eval_env"]["grayscale"] = args.grayscale
    config["eval_env"]["num_img_obs"] = args.framestack
    config['env']['imagesize'] = args.imagesize
    config['eval_env']['imagesize'] = args.imagesize
    config['env']['seed'] = args.seed
    config['eval_env']['seed'] = args.seed
    config["model"]["dueling"] = bool(args.dueling)
    config["algo"]["min_steps_learn"] = args.min_steps_learn
    config["algo"]["n_step_return"] = args.n_step
    config["algo"]["batch_size"] = args.batch_size
    config["algo"]["learning_rate"] = 0.0001
    config['algo']['replay_ratio'] = args.replay_ratio
    config['algo']['target_update_interval'] = args.target_update_interval
    config['algo']['target_update_tau'] = args.target_update_tau
    config['algo']['eps_steps'] = args.eps_steps
    config["algo"]["clip_grad_norm"] = args.max_grad_norm
    config['algo']['pri_alpha'] = 0.5
    config['algo']['pri_beta_steps'] = int(10e4)
    config['optim']['eps'] = 0.00015
    config["sampler"]["eval_max_trajectories"] = 100
    config["sampler"]["eval_n_envs"] = 100
    config["sampler"]["eval_max_steps"] = 100*28000  # 28k is just a safe ceiling
    config['sampler']['batch_B'] = args.batch_b
    config['sampler']['batch_T'] = args.batch_t

    config['agent']['eps_init'] = args.eps_init
    config['agent']['eps_final'] = args.eps_final
    config["model"]["noisy_nets_std"] = args.noisy_nets_std

    if args.noisy_nets:
        config['agent']['eps_eval'] = 0.001

    # New SPR Arguments
    config["model"]["imagesize"] = args.imagesize
    config["model"]["jumps"] = args.jumps
    config["model"]["dynamics_blocks"] = args.dynamics_blocks
    config["model"]["spr"] = args.spr
    config["model"]["noisy_nets"] = args.noisy_nets
    config["model"]["momentum_encoder"] = args.momentum_encoder
    config["model"]["shared_encoder"] = args.shared_encoder
    config["model"]["local_spr"] = args.local_spr
    config["model"]["global_spr"] = args.global_spr
    config["model"]["distributional"] = args.distributional
    config["model"]["renormalize"] = args.renormalize
    config["model"]["norm_type"] = args.norm_type
    config["model"]["augmentation"] = args.augmentation
    config["model"]["q_l1_type"] = args.q_l1_type
    config["model"]["dropout"] = args.dropout
    config["model"]["time_offset"] = args.time_offset
    config["model"]["aug_prob"] = args.aug_prob
    config["model"]["target_augmentation"] = args.target_augmentation
    config["model"]["eval_augmentation"] = args.eval_augmentation
    config["model"]["classifier"] = args.classifier
    config["model"]["final_classifier"] = args.final_classifier
    config['model']['momentum_tau'] = args.momentum_tau
    config["model"]["dqn_hidden_size"] = args.dqn_hidden_size
    config["model"]["model_rl"] = args.model_rl_weight
    config["model"]["residual_tm"] = args.residual_tm
    config["algo"]["model_rl_weight"] = args.model_rl_weight
    config["algo"]["reward_loss_weight"] = args.reward_loss_weight
    config["algo"]["model_spr_weight"] = args.model_spr_weight
    config["algo"]["t0_spr_loss_weight"] = args.t0_spr_loss_weight
    config["algo"]["time_offset"] = args.time_offset
    config["algo"]["distributional"] = args.distributional
    config["algo"]["delta_clip"] = args.delta_clip
    config["algo"]["prioritized_replay"] = args.prioritized_replay

    return config