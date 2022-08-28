from shutil import rmtree
from shutil import copyfile
from pathlib import Path
import inspect, optuna, os
from optuna.pruners import MedianPruner, HyperbandPruner, NopPruner
from optuna.samplers import PartialFixedSampler
from itertools import product
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm


import torch
from Xlearn.cli import Args, Const as C
from Xlearn.visual import write_analysis
from Xlearn.utils import pickle_dump, CloudpickleWrapper, color_str, timeit

@timeit(color='green', bold=True)
def run_experiment(run, config, seeds, log_dir, max_workers, chunksize=1, use_gpu=False, gpu_ids=None, copy_file=True, prune_exp=True):
    r"""A convenient function to parallelize the experiment (master-worker pipeline). 
    
    It is implemented by using `concurrent.futures.ProcessPoolExecutor`
    
    It automatically creates all subfolders for each pair of configuration and random seed
    to store the loggings of the experiment. The root folder is given by the user.
    Then all subfolders for each configuration are created with the name of their job IDs.
    Under each configuration subfolder, a set subfolders are created for each
    random seed (the random seed as folder name). Intuitively, an experiment could have 
    following directory structure::

        - logs
            - 0  # ID number
                - 123  # random seed
                - 345
                - 567
            - 1
                - 123
                - 345
                - 567
            - 2
                - 123
                - 345
                - 567
            - 3
                - 123
                - 345
                - 567
            - 4
                - 123
                - 345
                - 567
                
    Args:
        run (function): a function that defines an algorithm, it must take the 
            arguments `(config, seed, device, logdir)`
        config (Config): a :class:`Config` object defining all configuration settings
        seeds (list): a list of random seeds
        log_dir (str): a string to indicate the path to store loggings.
        max_workers (int): argument for ProcessPoolExecutor. if `None`, then all experiments run serially.
        chunksize (int): argument for Executor.map()
        use_gpu (bool): if `True`, then use CUDA. Otherwise, use CPU.
        gpu_ids (list): if `None`, then use all available GPUs. Otherwise, only use the
            GPU device defined in the list. 
    
    """
    check_valid_config(config)

    # extract config information
    KEYS: dict = config.obtain_key()        
    configs: list = config.make_configs(log_dir=log_dir)
    
    # create logging dir
    log_path: Path = Path(log_dir)
    if not log_path.exists():
        log_path.mkdir(parents=True)
    else:
        msg = f"Logging directory '{log_path.absolute()}' already existed, do you want to clean it ?"
        answer = ask_yes_or_no(msg)
        if answer:
            (sp := log_path/'source_files').exists() and rmtree(sp)
            (sp := log_path/'experiment.db').exists() and sp.unlink()
            log_path.mkdir(parents=True, exist_ok=True)
        else:  # back up
            import time
            current_time = time.strftime("%H:%M", time.localtime())
            (sp := log_path/'experiment.db').exists() and copyfile(
                sp, sp.with_name('%s:%s'%(current_time, sp.name))
            )
            print(f"The old database is copied to '{sp.absolute()}'. ")

    # save .py files from source folder
    if copy_file and not check_in_ipython():
        source_path = Path(inspect.getsourcefile(run)).parent
        target_path = log_path/'source_files/'
        copy_recursive(source_path, target_path)
        
        pickle_dump([
            {k: v for k,v in x.items() if k in list(KEYS)+['ID']} for x in configs
        ], log_path / 'configs', ext='.pkl')
    
    # Create unique id for each job
    jobs = list(enumerate(product(configs, seeds)))
    KEYS['seed'] = seeds
    storage = f"sqlite:///{str(log_path)}/experiment.db"
    study_name = config[C.EXP]
    write_analysis(log_path/'analysis.py', log_path, study_name, config[C.KEY])
    pruner = HyperbandPruner() if prune_exp else NopPruner()
    study = optuna.create_study(
        study_name=study_name,
        pruner=pruner,
        direction=config[C.DIR],
        storage=storage,
        load_if_exists = True
    )
    # save config to study
    for key in config:
        study._storage.set_study_user_attr(study._study_id, key, str(config[key]))
    
    # define how each job should be done, call (run) with parameter grid
    def _run(job, lock=None):
        job_id, (config, seed) = job
        # VERY IMPORTANT TO AVOID GETTING STUCK, oversubscription
        # see following links
        # https://github.com/pytorch/pytorch/issues/19163
        # https://software.intel.com/en-us/intel-threading-building-blocks-openmp-or-native-threads
        torch.set_num_threads(1)
        if use_gpu:
            num_gpu = torch.cuda.device_count()
            if gpu_ids is None:  # use all GPUs
                device_id = job_id % num_gpu
            else:
                assert all([i >= 0 and i < num_gpu for i in gpu_ids])
                device_id = gpu_ids[job_id % len(gpu_ids)]
            torch.cuda.set_device(device_id)
            device = torch.device(f'cuda:{device_id}')
        else:
            device = torch.device('cpu')
            
        print('\n\n')
        print(f'@ Experiment: ID: {config["ID"]} ({len(configs)}), Seed: {seed}, Device: {device}, Job: {job_id} ({len(jobs)}), PID: {os.getpid()}')
        print('#'*80)
        _args = Args(**config, lock=lock, device=device, seed=seed)
        _args.set('task_name', ' | '.join(f'{k}={_args.get(k)}' for k in {'exp_name': 0, **KEYS}))
        _args.set('job_id', job_id)
        pruner = HyperbandPruner() if prune_exp else NopPruner()
        study = optuna.load_study(study_name=study_name, storage=storage, pruner=pruner)

        def check_type(vs: list):
            return sorted(vs), isinstance(vs[0], str)

        def objective(trial):
            for k, vs in KEYS.items():
                vs, str_flag = check_type(vs)
                if str_flag:
                    _ = trial.suggest_categorical(k, vs)
                else:
                    _ = trial.suggest_uniform(k, vs[0], vs[-1])
            _args.set('trial', trial)
            result = run(_args)
            if use_gpu: torch.cuda.empty_cache()
            return result

        fixed_params = {k: _args.get(k) for k in KEYS}
        study.sampler = PartialFixedSampler(fixed_params, study.sampler)
        study.optimize(objective, n_trials=1, gc_after_trial=True)
    
    if max_workers is None:
        results = [_run(job, None) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(max_workers, len(jobs))) as executor:
            l = mp.Manager().Lock()
            n = len(jobs)
            results = list(tqdm(executor.map(CloudpickleWrapper(_run), jobs, [l]*n, chunksize=chunksize), total=n))
    print(color_str(f'\nExperiment finished. Loggings are stored in {log_path.absolute()}. ', 'cyan', bold=True))
    return results


def copy_recursive(source: Path, target: Path): 
    target.mkdir(exist_ok=True, parents=True)
    for item in source.iterdir():
        if item.is_dir() and not item.name.startswith('__'):
            new_target = target/item.name
            new_target.mkdir(exist_ok=True)
            copy_recursive(item, new_target)
        else:
            for file in source.glob('*.py'):
                copyfile(file, target/file.name)

def check_in_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

def check_valid_config(config):
    assert C.EXP in config, "you have to specify an `exp_name`"
    assert C.KEY in config, "you have to specify `optim_metric` used for model comparisons"
    assert C.DIR in config, "you have to specify `optim_direction` to which better result is defined"

def ask_yes_or_no(msg):
    r"""Ask user to enter yes or no to a given message. 
    msg (str): a message
    """
    print(msg)
    while True:
        answer = str(input('>>> ')).lower().strip()
        if answer[0] == 'y':
            return True
        elif answer[0] == 'n':
            return False
        else:
            print("Please answer 'yes' or 'no':")
    


