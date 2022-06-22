import os
from multiprocessing import set_start_method

import train_distributed as distributed
import train_pbt as pbt
import train_sequential as sequential
import train_td3_cemrl as cemrl
import train_td3_dvd as dvd

from fastpbrl.replay_buffer import ReplayBufferConfig

# Since we are running multiple training runs in a row, we
# need jax to de-allocate memory between tests otherwise
# we can run into out-of-memory errors.
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def test_sequential():
    algorithm_name_to_hyperparams = {
        "SAC": sequential.SACHyperParams(),
        "TD3": sequential.TD3HyperParams(),
    }
    for name, hyperparams in algorithm_name_to_hyperparams.items():
        print(f"\nTest {name} Sequential \n\n")
        config = sequential.SequentialConfig(
            hyperparams=hyperparams,
            total_num_env_steps=2000,
            num_warmup_env_steps=100,
            min_size_to_sample=100,
        )
        assert sequential.main(config) == 0


def test_distributed():
    algorithm_name_to_hyperparams = {
        "SAC": sequential.SACHyperParams(),
        "TD3": sequential.TD3HyperParams(),
    }
    for name, hyperparams in algorithm_name_to_hyperparams.items():
        print(f"\nTest {name} Distributed \n\n")
        config = distributed.DistributedConfig(
            hyperparams=hyperparams,
            num_actors=2,
            total_num_env_steps=3000,
            replay_buffer_config=ReplayBufferConfig(
                samples_per_insert=256.0,
                insert_error_buffer=100.0,
                min_size_to_sample=100,
            ),
        )
        assert distributed.main(config) == 0


def test_pbt():
    algorithm_name_to_agent = {
        "SAC": pbt.SACPBT,
        "TD3": pbt.TD3PBT,
    }
    for name, agent in algorithm_name_to_agent.items():
        print(f"\nTest {name} PBT \n\n")
        config = pbt.PBTConfig(
            agent=agent,
            population_size=4,
            num_devices=1,
            num_actor_processes_per_device=2,
            total_num_env_steps=20_000,
            pbt_update_frequency=10_000,
            num_warmup_env_steps=1_000,
            evaluation_frequency=5_000,
            replay_buffer_config=ReplayBufferConfig(
                samples_per_insert=256.0,
                insert_error_buffer=100.0,
                min_size_to_sample=100,
            ),
        )
        assert pbt.main(config) == 0


def test_dvd():
    print("\nTest TD3 DvD \n\n")
    config = dvd.DVDTD3Config(
        population_size=3,
        total_num_env_steps=1000,
        num_warmup_env_steps=100,
        replay_buffer_config=ReplayBufferConfig(
            samples_per_insert=256.0,
            insert_error_buffer=100.0,
            min_size_to_sample=100,
        ),
    )
    assert dvd.main(config) == 0


def test_cemrl():
    print("\nTest TD3 CEM-RL \n\n")
    config = cemrl.CEMRLTD3Config(
        population_size=4,
        cem_parameters=cemrl.CrossEntropyParameters(
            num_elites=2,
        ),
        total_num_env_steps=1000,
        num_warmup_env_steps=1,
    )
    assert cemrl.main(config) == 0


if __name__ == "__main__":
    set_start_method("spawn")
    test_sequential()
    test_distributed()
    test_pbt()
    test_dvd()
    test_cemrl()
