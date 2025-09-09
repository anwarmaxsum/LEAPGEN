## This is repository for LEAPGen from our paper entitled VISION AND LANGUAGE SYNERGY FOR REHEARSAL FREE CONTINUAL LEARNING [ICLR 2025]


#Example Running Script

python main_leapgen.py cifar100_leapgen --length 30 --epochs 3 --num_tasks 5 --k_mul 50 --lr 0.01 

python main_leapgen.py cifar100_leapgen --length 30 --epochs 10 --num_tasks 10 --k_mul 50 --lr 0.01

python main_leapgen.py cifar100_leapgen --length 30 --epochs 10 --num_tasks 20 --k_mul 50 --lr 0.01 


python main_leapgen.py imr_leapgen --length 30 --epochs 5 --num_tasks 5 --k_mul 1.0 --lr 0.05 --intertask_coeff 0.1 

python main_leapgen.py imr_leapgen --length 30 --epochs 10 --num_tasks 10 --k_mul 1.0 --lr 0.05 --intertask_coeff 0.1

python main_leapgen.py imr_leapgen --length 30 --epochs 20 --num_tasks 20 --k_mul 1.0 --lr 0.05 --intertask_coeff 0.1


python main_leapgen.py cub_leapgen --length 30 --epochs 20 --num_tasks 10 --k_mul 200 --lr 0.005 --intertask_coeff 0.1
