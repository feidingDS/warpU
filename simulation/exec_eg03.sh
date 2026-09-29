#!/bin/bash

# setup with variational inference
python gmmvi-gauss.py --setID 3 --theta_d 30
python gmmvi-gauss.py --setID 3 --theta_d 30 --phimixSubID 2
python gmmvi-neural_v2.py --setID 3 --theta_d 30 --phimix_type 1


# sampling
python -u batchSampling.py --setID 3 --theta_d 30 --startI 0 --totalRep 1 --phimixType 1 
python -u batchSampling.py --setID 3 --theta_d 30 --startI 0 --totalRep 1 --phimixType 2 
python -u HM_Indep.py --setID 3 --theta_d 30 --startI 0 --totalRep 100 --phimixType 2 
python -u batchSampling.py --setID 3 --theta_d 30 --startI 0 --totalRep 1 --phimixType 4 

# bridge estimation
python -u bridgeEstimator.py --setID 3 --theta_d 30 --phimix_type 1 --sample_type 1  --parallel 1
python -u bridgeEstimator.py --setID 3 --theta_d 30 --phimix_type 1 --sample_type 2  --parallel 1
python -u bridgeEstimator.py --setID 3 --theta_d 30 --phimix_type 1 --sample_type 5  --parallel 1
python -u bridgeEstimator.py --setID 3 --theta_d 30 --phimix_type 1 --sample_type 7  --parallel 1

python -u bridgeEstimator.py --setID 3 --theta_d 30 --phimix_type 2 --sample_type 1  --parallel 1
python -u bridgeEstimator.py --setID 3 --theta_d 30 --phimix_type 2 --sample_type 2  --parallel 1
python -u bridgeEstimator.py --setID 3 --theta_d 30 --phimix_type 2 --sample_type 5  --parallel 1
python -u bridgeEstimator.py --setID 3 --theta_d 30 --phimix_type 2 --sample_type 7  --parallel 1

python -u bridgeEstimator.py --setID 3 --theta_d 30 --phimix_type 4 --sample_type 1  --parallel 0
python -u bridgeEstimator.py --setID 3 --theta_d 30 --phimix_type 4 --sample_type 2  --parallel 0
python -u bridgeEstimator.py --setID 3 --theta_d 30 --phimix_type 4 --sample_type 5  --parallel 0
python -u bridgeEstimator.py --setID 3 --theta_d 30 --phimix_type 4 --sample_type 7  --parallel 0


python -u LAIS_Estimator.py --setID 3 --theta_d 30 --phimix_type 1 --totalRep 100 --samplerType 1
python -u LAIS_Estimator.py --setID 3 --theta_d 30 --phimix_type 2 --totalRep 100 --samplerType 1
python -u LAIS_Estimator.py --setID 3 --theta_d 30 --phimix_type 4 --totalRep 100 --samplerType 1
python -u LAIS_Estimator.py --setID 3 --theta_d 30 --phimix_type 2 --totalRep 100 --samplerType 2
