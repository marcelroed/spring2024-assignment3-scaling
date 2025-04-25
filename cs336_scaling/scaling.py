from pathlib import Path
import requests
from json import dumps, loads
from random import randint, randrange
import numpy as np
import pandas as pd

API_KEY = 'ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQDqFuvp5DTQ+5jJxkmu/P8K6XKNEuALYKr97Jv47RlmvAs/DI/YUk4Xklz4m2F2zbN+pGFdwe2B1AbROTAfo7qTv08ar1ZLCPlKDfvgH9eUMEMDUCsZURwCYrukbphw6wnLrhPIKs2i925xWVDGVuSfKgK4X1rlqdIrUbhBSyTWWEoZL/d9yKy3SRqFEdp96ecMJ6oKzSb58vjstGsv+YarbitzENtyfW2r24V2qySXpPtuo6DGAwZcI8fnAwPwZpj8j6umauYb7YJHW5PQzpufo4fzn02hvi3g82s8qXSDthKlSB8r1GsIB4QD/+arY5Pnniich541+I4dGZ62m0d7FQD9osOyyTGn+SDq1nhDsaeSLSJm1sx4gjJKmUdyv4NeFH5nlfZEQqAeTOu5fjG4V45eq8gvppMSq8RftaMhZ7G5QZej5SIlbwIof32LFjIg0mup9tHTbBXH/uODkZKVkArtcUpXtQQV8gzGcySlDvrJUb8KspEAnEPAzcClJuluYDvxnkpMxSGroB8a0O7kCYgR5VBu3/Pr9aY6h4xsp0FuvtGAMrX1Qa71WprSqAbr50qh/0Y3OK6+NNsG5t3em3xZO5ntdF0CrEjKFBOg3aK9Z7PiPulNV/MLNCX2eq3szA/vn7ubFsA0hGFUkvtUPbVYsCB/0YegXS/VTvN6UQ== marcel@Marcels-MacBook-Pro.home'

result_file = Path('results/results.txt')


def append_result(result):
    with result_file.open('a') as f:
        f.write(dumps(result) + '\n')


def submit_with_config(*, d_model, num_layers, num_heads, batch_size, learning_rate, train_flops):
    config = {'d_model': d_model, 'num_layers': num_layers, 'num_heads': num_heads, 'batch_size': batch_size, 'learning_rate': learning_rate, 'train_flops':
train_flops, 'api_key': API_KEY}
    print('Going to get input')
    # if input('Are you sure you want to submit? (y/n) ') != 'y':
    #     print('Submission cancelled')
    #     return None
    result: dict = requests.get('http://tahoma.stanford.edu:8000/loss', config).json()
    config.pop('api_key')
    result.update(config)
    append_result(result)
    return result


def get_total_flops_used():
    config = {'api_key': API_KEY}
    result = requests.get('http://tahoma.stanford.edu:8000/total_flops_used', config).json()
    append_result(result)
    return result

def get_previous_runs():
    lines = []
    with open('results/results.txt', 'r') as f:
        for line in f:
            line = loads(line)
            lines.append(line)
    points = [p for p in lines if isinstance(p, dict) and 'message' not in p]
    df = pd.DataFrame(points)
    return df


def do_random_isoflop(n_parameters):
    # n_parameters = 12 * n_layer * d_model^2

    d_model = (2**(np.log2(256) + np.random.rand(*n_parameters.shape) * (np.log2(1024) - np.log2(256)))).astype(np.int64)
    num_heads = d_model // 128
    d_model = num_heads * 128
    num_layers = (n_parameters / (12 * d_model**2)).astype(np.int64)
    n_parameters = 12 * num_layers * d_model**2

    df = pd.DataFrame({'d_model': d_model, 'num_layers': num_layers, 'n_parameters': n_parameters, 'num_heads': num_heads})

    # Remove rows that contain zeros
    df = df[(df != 0).all(1) & (df['num_layers'] >= 2) & (df['num_layers'] <= 24) & (df['num_heads'] >= 2)].drop_duplicates()
    return df


def sample_to_do(from_p, to_p):
    df = None
    for p in np.geomspace(from_p, to_p, 100):
        new_df = do_random_isoflop(np.full(1000000, p))
        if df is None:
            df = new_df
        else:
            df = pd.concat([df, new_df])
            # df = df.reindex()
            df = df.drop_duplicates()
    lr = np.geomspace(1e-4, 1e-3, 10)[np.random.randint(0, 10, len(df))]
    df['learning_rate'] = lr
    print(df)
    return df

# def run_if_not_ran(df):



if __name__ == '__main__':
    df = sample_to_do(1e6, 1e8)
    df['batch_size'] = 128
    df['train_flops'] = int(1e14)
    print(df)
    df = df.sample(1)
    for d in df.to_dict(orient='records'):
        print(d)
        d.pop('n_parameters')
        print(submit_with_config(**d))
