#!/bin/bash

PYTHONPATH=. python run_m4.py --dataset 'Hourly' --directory /pos/esrnn_of --gpu_id 0 --use_cpu 1 2>&1 | tee ./results/hourly_cpu_results.log

PYTHONPATH=. python run_m4.py --dataset 'Weekly' --directory /pos/esrnn_of --gpu_id 0 --use_cpu 1 2>&1 | tee ./results/weekly_cpu_results.log

PYTHONPATH=. python run_m4.py --dataset 'Daily' --directory /pos/esrnn_of --gpu_id 0 --use_cpu 1 2>&1 | tee ./results/daily_cpu_results.log

PYTHONPATH=. python run_m4.py --dataset 'Yearly' --directory /pos/esrnn_of --gpu_id 0 --use_cpu 1 2>&1 | tee ./results/yearly_cpu_results.log

PYTHONPATH=. python run_m4.py --dataset 'Quarterly' --directory /pos/esrnn_of --gpu_id 0 --use_cpu 1 2>&1 | tee ./results/quarterly_cpu_results.log

PYTHONPATH=. python run_m4.py --dataset 'Monthly' --directory /pos/esrnn_of --gpu_id 0 --use_cpu 1 2>&1 | tee ./results/monthly_cpu_results.log
