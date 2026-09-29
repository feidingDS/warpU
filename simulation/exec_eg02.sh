#!/bin/bash
RepTotal=100
simuID=2
ST=0

python -u gmmvi-gauss.py --setID 2 --theta_d 30 
python -u gmmvi-skew.py --setID 2 --theta_d 30 
python -u gmmvi-gamma.py --setID 2 --theta_d 30 

# warpu sampling
python -u warpuSampling.py --setID $simuID --phimix 1 --theta_d 30 --totalRep $RepTotal --startI $ST 
python -u warpuSampling.py --setID $simuID --phimix 2 --theta_d 30 --totalRep $RepTotal --startI $ST 
python -u warpuSampling.py --setID $simuID --phimix 3 --theta_d 30 --totalRep $RepTotal --startI $ST 

# parallel sampling
python -u pt_tuning.py --setID $simuID --vanilla 0 --theta_d 30  --samplerType 0
python -u pt_tuning.py --setID $simuID --vanilla 0 --theta_d 30  --samplerType 1
python -u ptSampling.py --setID $simuID --vanilla 0 --theta_d 30 --totalRep $RepTotal --samplerType 0  --startI $ST 
python -u ptSampling.py --setID $simuID --vanilla 0 --theta_d 30 --totalRep $RepTotal --samplerType 1 --startI $ST  

# bridge estimation
python -u bridgeEstimator.py --setID 2 --theta_d 30 --phimix_type 1 --sample_type 1 --parallel 1
python -u bridgeEstimator.py --setID 2 --theta_d 30 --phimix_type 2 --sample_type 1 --parallel 1
python -u bridgeEstimator.py --setID 2 --theta_d 30 --phimix_type 3 --sample_type 1 --parallel 1
python -u bridgeEstimator.py --setID 2 --theta_d 30 --phimix_type 1 --sample_type 2 --parallel 1
python -u bridgeEstimator.py --setID 2 --theta_d 30 --phimix_type 2 --sample_type 2 --parallel 1
python -u bridgeEstimator.py --setID 2 --theta_d 30 --phimix_type 3 --sample_type 2 --parallel 1
python -u bridgeEstimator.py --setID 2 --theta_d 30 --phimix_type 1 --sample_type 3 --parallel 1
python -u bridgeEstimator.py --setID 2 --theta_d 30 --phimix_type 2 --sample_type 3 --parallel 1
python -u bridgeEstimator.py --setID 2 --theta_d 30 --phimix_type 3 --sample_type 3 --parallel 1
python -u bridgeEstimator.py --setID 2 --theta_d 30 --phimix_type 1 --sample_type 6 --parallel 1
python -u bridgeEstimator.py --setID 2 --theta_d 30 --phimix_type 2 --sample_type 6 --parallel 1
python -u bridgeEstimator.py --setID 2 --theta_d 30 --phimix_type 3 --sample_type 6 --parallel 1
python -u bridgeEstimator.py --setID 2 --theta_d 30 --phimix_type 1 --sample_type 7 --parallel 1
python -u bridgeEstimator.py --setID 2 --theta_d 30 --phimix_type 2 --sample_type 7 --parallel 1
python -u bridgeEstimator.py --setID 2 --theta_d 30 --phimix_type 3 --sample_type 7 --parallel 1