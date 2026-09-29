#!/bin/bash
RepTotal=40
simuID=1
ST=0

python -u pt_tuning.py --setID $simuID --vanilla 1 --theta_d 10  --samplerType 0 
python -u pt_tuning.py --setID $simuID --vanilla 1 --theta_d 50  --samplerType 0 
python -u pt_tuning.py --setID $simuID --vanilla 1 --theta_d 100  --samplerType 0 
python -u pt_tuning.py --setID $simuID --vanilla 1 --theta_d 500  --samplerType 0 
python -u pt_tuning.py --setID $simuID --vanilla 1 --theta_d 1000  --samplerType 0 
python -u ptSampling.py --setID $simuID --vanilla 1 --theta_d 10 --totalRep $RepTotal --samplerType 0  --startI $ST 
python -u ptSampling.py --setID $simuID --vanilla 1 --theta_d 50 --totalRep $RepTotal --samplerType 0  --startI $ST 
python -u ptSampling.py --setID $simuID --vanilla 1 --theta_d 100 --totalRep $RepTotal --samplerType 0  --startI $ST 
python -u ptSampling.py --setID $simuID --vanilla 1 --theta_d 500 --totalRep $RepTotal --samplerType 0  --startI $ST 
python -u ptSampling.py --setID $simuID --vanilla 1 --theta_d 1000 --totalRep $RepTotal --samplerType 0 --startI $ST  


python -u pt_tuning.py --setID $simuID --vanilla 1 --theta_d 10  --samplerType 1 
python -u pt_tuning.py --setID $simuID --vanilla 1 --theta_d 50  --samplerType 1
python -u pt_tuning.py --setID $simuID --vanilla 1 --theta_d 100  --samplerType 1 
python -u pt_tuning.py --setID $simuID --vanilla 1 --theta_d 500  --samplerType 1
python -u pt_tuning.py --setID $simuID --vanilla 1 --theta_d 1000  --samplerType 1
python -u ptSampling.py --setID $simuID --vanilla 1 --theta_d 10 --totalRep $RepTotal --samplerType 1  --startI $ST 
python -u ptSampling.py --setID $simuID --vanilla 1 --theta_d 50 --totalRep $RepTotal --samplerType 1  --startI $ST 
python -u ptSampling.py --setID $simuID --vanilla 1 --theta_d 100 --totalRep $RepTotal --samplerType 1  --startI $ST 
python -u ptSampling.py --setID $simuID --vanilla 1 --theta_d 500 --totalRep $RepTotal --samplerType 1  --startI $ST 
python -u ptSampling.py --setID $simuID --vanilla 1 --theta_d 1000 --totalRep $RepTotal --samplerType 1 --startI $ST  


python -u pt_tuning.py --setID $simuID --vanilla 0 --theta_d 10  --samplerType 0 
python -u pt_tuning.py --setID $simuID --vanilla 0 --theta_d 50  --samplerType 0 
python -u pt_tuning.py --setID $simuID --vanilla 0 --theta_d 100  --samplerType 0 
python -u pt_tuning.py --setID $simuID --vanilla 0 --theta_d 500  --samplerType 0 
python -u pt_tuning.py --setID $simuID --vanilla 0 --theta_d 1000  --samplerType 0 
python -u ptSampling.py --setID $simuID --vanilla 0 --theta_d 10 --totalRep $RepTotal --samplerType 0  --startI $ST 
python -u ptSampling.py --setID $simuID --vanilla 0 --theta_d 50 --totalRep $RepTotal --samplerType 0 --startI $ST  
python -u ptSampling.py --setID $simuID --vanilla 0 --theta_d 100 --totalRep $RepTotal --samplerType 0 --startI $ST  
python -u ptSampling.py --setID $simuID --vanilla 0 --theta_d 500 --totalRep $RepTotal --samplerType 0 --startI $ST  
python -u ptSampling.py --setID $simuID --vanilla 0 --theta_d 1000 --totalRep $RepTotal --samplerType 0 --startI $ST  


python -u pt_tuning.py --setID $simuID --vanilla 0 --theta_d 10  --samplerType 1
python -u pt_tuning.py --setID $simuID --vanilla 0 --theta_d 50  --samplerType 1 
python -u pt_tuning.py --setID $simuID --vanilla 0 --theta_d 100  --samplerType 1 
python -u pt_tuning.py --setID $simuID --vanilla 0 --theta_d 500  --samplerType 1 
python -u pt_tuning.py --setID $simuID --vanilla 0 --theta_d 1000  --samplerType 1
python -u ptSampling.py --setID $simuID --vanilla 0 --theta_d 10 --totalRep $RepTotal --samplerType 1 --startI $ST  
python -u ptSampling.py --setID $simuID --vanilla 0 --theta_d 50 --totalRep $RepTotal --samplerType 1 --startI $ST  
python -u ptSampling.py --setID $simuID --vanilla 0 --theta_d 100 --totalRep $RepTotal --samplerType 1 --startI $ST  
python -u ptSampling.py --setID $simuID --vanilla 0 --theta_d 500 --totalRep $RepTotal --samplerType 1 --startI $ST  
python -u ptSampling.py --setID $simuID --vanilla 0 --theta_d 1000 --totalRep $RepTotal --samplerType 1 --startI $ST  
