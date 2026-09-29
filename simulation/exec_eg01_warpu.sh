#!/bin/bash
RepTotal=40
simuID=1
ST=0

python -u gmmvi-gauss.py --setID 1 --theta_d 10 
python -u gmmvi-gauss.py --setID 1 --theta_d 50 
python -u gmmvi-gauss.py --setID 1 --theta_d 100 
python -u gmmvi-gauss.py --setID 1 --theta_d 500 
python -u gmmvi-gauss.py --setID 1 --theta_d 1000 

python -u gmmvi-skew.py --setID 1 --theta_d 10 
python -u gmmvi-skew.py --setID 1 --theta_d 50 
python -u gmmvi-skew.py --setID 1 --theta_d 100 
python -u gmmvi-skew.py --setID 1 --theta_d 500
python -u gmmvi-skew.py --setID 1 --theta_d 1000 


python -u gmmvi-gamma.py --setID 1 --theta_d 10 
python -u gmmvi-gamma.py --setID 1 --theta_d 50 
python -u gmmvi-gamma.py --setID 1 --theta_d 100 
python -u gmmvi-gamma.py --setID 1 --theta_d 500 
python -u gmmvi-gamma.py --setID 1 --theta_d 1000 

python -u warpuSampling.py --setID $simuID --phimix 1 --theta_d 10 --totalRep $RepTotal --startI $ST 
python -u warpuSampling.py --setID $simuID --phimix 1 --theta_d 50 --totalRep $RepTotal --startI $ST 
python -u warpuSampling.py --setID $simuID --phimix 1 --theta_d 100 --totalRep $RepTotal --startI $ST 
python -u warpuSampling.py --setID $simuID --phimix 1 --theta_d 500 --totalRep $RepTotal --startI $ST 
python -u warpuSampling.py --setID $simuID --phimix 1 --theta_d 1000 --totalRep $RepTotal --startI $ST 

python -u warpuSampling.py --setID $simuID --phimix 2 --theta_d 10 --totalRep $RepTotal --startI $ST 
python -u warpuSampling.py --setID $simuID --phimix 2 --theta_d 50 --totalRep $RepTotal --startI $ST 
python -u warpuSampling.py --setID $simuID --phimix 2 --theta_d 100 --totalRep $RepTotal --startI $ST 
python -u warpuSampling.py --setID $simuID --phimix 2 --theta_d 500 --totalRep $RepTotal --startI $ST 
python -u warpuSampling.py --setID $simuID --phimix 2 --theta_d 1000 --totalRep $RepTotal --startI $ST 

python -u warpuSampling.py --setID $simuID --phimix 3 --theta_d 10 --totalRep $RepTotal --startI $ST 
python -u warpuSampling.py --setID $simuID --phimix 3 --theta_d 50 --totalRep $RepTotal --startI $ST 
python -u warpuSampling.py --setID $simuID --phimix 3 --theta_d 100 --totalRep $RepTotal --startI $ST 
python -u warpuSampling.py --setID $simuID --phimix 3 --theta_d 500 --totalRep $RepTotal --startI $ST 
python -u warpuSampling.py --setID $simuID --phimix 3 --theta_d 1000 --totalRep $RepTotal --startI $ST 
