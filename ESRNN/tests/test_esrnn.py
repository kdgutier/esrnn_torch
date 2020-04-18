#Testing ESRNN
import runpy
import os

print('\n')
print(10*'='+'TEST ESRNN'+10*'=')
print('\n')

def test_esrnn_hourly():
    if not os.path.exists('./data'):
        os.mkdir('./data')

    print('\n')
    print(10*'='+'HOURLY'+10*'=')
    print('\n')

    exec_str = 'python -m ESRNN.m4_run --dataset Hourly '
    exec_str += '--results_directory ./data --gpu_id 0 '
    exec_str += '--use_cpu 1 --num_obs 100 --test 1'
    results = os.system(exec_str)

    if results==0:
        print('Test completed')
    else:
        raise Exception('Something went wrong')

def test_esrnn_weekly():
    if not os.path.exists('./data'):
        os.mkdir('./data')

    print('\n')
    print(10*'='+'WEEKLY'+10*'=')
    print('\n')

    exec_str = 'python -m ESRNN.m4_run --dataset Weekly '
    exec_str += '--results_directory ./data --gpu_id 0 '
    exec_str += '--use_cpu 1 --num_obs 100 --test 1'
    results = os.system(exec_str)

    if results==0:
        print('Test completed')
    else:
        raise Exception('Something went wrong')


def test_esrnn_daily():
    if not os.path.exists('./data'):
        os.mkdir('./data')

    print('\n')
    print(10*'='+'DAILY'+10*'=')
    print('\n')

    exec_str = 'python -m ESRNN.m4_run --dataset Daily '
    exec_str += '--results_directory ./data --gpu_id 0 '
    exec_str += '--use_cpu 1 --num_obs 100 --test 1'
    results = os.system(exec_str)

    if results==0:
        print('Test completed')
    else:
        raise Exception('Something went wrong')


def test_esrnn_monthly():
    if not os.path.exists('./data'):
        os.mkdir('./data')


    print('\n')
    print(10*'='+'MONTHLY'+10*'=')
    print('\n')

    exec_str = 'python -m ESRNN.m4_run --dataset Monthly '
    exec_str += '--results_directory ./data --gpu_id 0 '
    exec_str += '--use_cpu 1 --num_obs 100 --test 1'
    results = os.system(exec_str)

    if results==0:
        print('Test completed')
    else:
        raise Exception('Something went wrong')


def test_esrnn_quarterly():
    if not os.path.exists('./data'):
        os.mkdir('./data')

    print('\n')
    print(10*'='+'QUARTERLY'+10*'=')
    print('\n')

    exec_str = 'python -m ESRNN.m4_run --dataset Quarterly '
    exec_str += '--results_directory ./data --gpu_id 0 '
    exec_str += '--use_cpu 1 --num_obs 100 --test 1'
    results = os.system(exec_str)

    if results==0:
        print('Test completed')
    else:
        raise Exception('Something went wrong')


def test_esrnn_yearly():
    if not os.path.exists('./data'):
        os.mkdir('./data')

    print('\n')
    print(10*'='+'YEARLY'+10*'=')
    print('\n')

    exec_str = 'python -m ESRNN.m4_run --dataset Yearly '
    exec_str += '--results_directory ./data --gpu_id 0 '
    exec_str += '--use_cpu 1 --num_obs 100 --test 1'
    results = os.system(exec_str)

    if results==0:
        print('Test completed')
    else:
        raise Exception('Something went wrong')
