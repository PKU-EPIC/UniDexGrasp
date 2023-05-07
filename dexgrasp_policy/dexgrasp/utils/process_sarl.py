def process_dagger_value(args, env, cfg_train, logdir):
    from algorithms.rl.dagger_value import DAGGERVALUE, Actor, ActorCritic, ActorCriticDagger
    learn_cfg = cfg_train["learn"]
    #is_testing = learn_cfg["test"]
    is_testing = args.test
    is_vision = args.vision
    # is_testing = True
    # Override resume and testing flags if they are passed as parameters.
    if args.model_dir != "":
        #is_testing = True
        chkpt_path = args.model_dir
    expert_chkpt_path = ""
    if args.expert_model_dir != "":
        #is_testing = True
        expert_chkpt_path = args.expert_model_dir

    logdir = logdir + "_seed{}".format(env.task.cfg["seed"])
    """Set up the DAgger value system for training or inferencing."""
    dagger_value = DAGGERVALUE(vec_env=env,
              actor_class=Actor,
              actor_critic_class=ActorCriticDagger,
              actor_critic_class_expert=ActorCritic,
              num_transitions_per_env=learn_cfg["nsteps"],
              num_learning_epochs=learn_cfg["noptepochs"],
              num_mini_batches=learn_cfg["nminibatches"],
              buffer_size=learn_cfg["buffer_size"],
              init_noise_std=learn_cfg.get("init_noise_std", 0.3),
              schedule=learn_cfg.get("schedule", "fixed"),
              model_cfg=cfg_train["policy"],
              device=env.rl_device,
              sampler=learn_cfg.get("sampler", 'sequential'),
              log_dir=logdir,
              is_testing=is_testing,
              print_log=learn_cfg["print_log"],
              apply_reset=False,
              asymmetric=(env.num_states > 0),
              expert_chkpt_path = expert_chkpt_path,
              is_vision = is_vision
              )

    if is_testing and args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        dagger_value.test(chkpt_path)
    elif args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        dagger_value.load(chkpt_path)

    return dagger_value

def process_ppo(args, env, cfg_train, logdir):
    from algorithms.rl.ppo import PPO, ActorCritic
    learn_cfg = cfg_train["learn"]
    #is_testing = learn_cfg["test"]
    is_testing = args.test
    is_vision = args.vision
    # is_testing = True
    # Override resume and testing flags if they are passed as parameters.
    if args.model_dir != "":
        #is_testing = True
        chkpt_path = args.model_dir

    logdir = logdir + "_seed{}".format(env.task.cfg["seed"])
    """Set up the PPO system for training or inferencing."""
    ppo = PPO(vec_env=env,
              actor_critic_class=ActorCritic,
              num_transitions_per_env=learn_cfg["nsteps"],
              num_learning_epochs=learn_cfg["noptepochs"],
              num_mini_batches=learn_cfg["nminibatches"],
              clip_param=learn_cfg["cliprange"],
              gamma=learn_cfg["gamma"],
              lam=learn_cfg["lam"],
              init_noise_std=learn_cfg.get("init_noise_std", 0.3),
              value_loss_coef=learn_cfg.get("value_loss_coef", 2.0),
              entropy_coef=learn_cfg["ent_coef"],
              learning_rate=learn_cfg["optim_stepsize"],
              max_grad_norm=learn_cfg.get("max_grad_norm", 2.0),
              use_clipped_value_loss=learn_cfg.get("use_clipped_value_loss", False),
              schedule=learn_cfg.get("schedule", "fixed"),
              desired_kl=learn_cfg.get("desired_kl", None),
              model_cfg=cfg_train["policy"],
              device=env.rl_device,
              sampler=learn_cfg.get("sampler", 'sequential'),
              log_dir=logdir,
              is_testing=is_testing,
              print_log=learn_cfg["print_log"],
              apply_reset=False,
              asymmetric=(env.num_states > 0),
              is_vision=is_vision,
              )

    if is_testing and args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        ppo.test(chkpt_path)
    elif args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        ppo.load(chkpt_path)

    return ppo

def process_dagger(args, env, cfg_train, logdir):
    from algorithms.rl.dagger import DAGGER, Actor, ActorCritic
    learn_cfg = cfg_train["learn"]
    #is_testing = learn_cfg["test"]
    is_testing = args.test
    is_vision = args.vision
    # is_testing = True
    # Override resume and testing flags if they are passed as parameters.
    if args.model_dir != "":
        #is_testing = True
        chkpt_path = args.model_dir
    expert_chkpt_path = ""
    if args.expert_model_dir != "":
        #is_testing = True
        expert_chkpt_path = args.expert_model_dir

    logdir = logdir + "_seed{}".format(env.task.cfg["seed"])
    """Set up the DAgger system for training or inferencing."""
    dagger = DAGGER(vec_env=env,
              actor_class=Actor,
              actor_critic_class=ActorCritic,
              num_transitions_per_env=learn_cfg["nsteps"],
              num_learning_epochs=learn_cfg["noptepochs"],
              num_mini_batches=learn_cfg["nminibatches"],
              buffer_size=learn_cfg["buffer_size"],
              init_noise_std=learn_cfg.get("init_noise_std", 0.3),
              schedule=learn_cfg.get("schedule", "fixed"),
              model_cfg=cfg_train["policy"],
              device=env.rl_device,
              sampler=learn_cfg.get("sampler", 'sequential'),
              log_dir=logdir,
              is_testing=is_testing,
              print_log=learn_cfg["print_log"],
              apply_reset=False,
              asymmetric=(env.num_states > 0),
              expert_chkpt_path = expert_chkpt_path,
              is_vision = is_vision
              )

    if is_testing and args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        dagger.test(chkpt_path)
    elif args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        dagger.load(chkpt_path)

    return dagger
