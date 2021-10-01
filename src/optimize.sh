#!/bin/bash
#SBATCH -p medium
#SBATCH -o /usr/users/jmeyer3/ownCloud/hpc_outfiles/outfile-%J #./../Results/hpc-outfiles/outfile-%J
#SBATCH -N 1
#SBATCH -n 24 #multiples of 24
#SBATCH -t 2-0:00:00
#S#BATCH --mail-type=ALL
#SBATCH --mail-user=jmeyer3@gwdg.de
#SBATCH --mem=300G
#S#BATCH --qos=short
#SBATCH -C scratch

only_transformer='True' #'False' 
simple_test='True' #'False'
regression='True' #'True'
optimizer='Random' #'Bayes'

start=$(date +%s)
echo $date
script=Optimizer.py 

cmd="python3 $script"
cmd="$cmd --simple_test $simple_test"
cmd="$cmd --only_transformer $only_transformer"
cmd="$cmd --regression $regression"
cmd="$cmd --regression $regression"
cmd="$cmd --optimizer $optimizer"
cmd="$cmd --max_trials 250"
cmd="$cmd --num_initial_points 2" # != 1   !!! 
cmd="$cmd --beta 2.7" #Check kt version!
cmd="$cmd --default_nodes 5000"
cmd="$cmd --default_leaky 1.0"
cmd="$cmd --default_kappa 3"
cmd="$cmd --default_radius 0.6"
cmd="$cmd --default_sigma 0.2"
cmd="$cmd --default_log_2_n_esn 0"
cmd="$cmd --default_overlay_ratio 0.025"

echo $cmd
eval $cmd

finnish=$(date +%s)

diff=$(( ( finnish - start ) )) #$SECONDS

echo "$(($diff / 3600)) hours, $((($diff / 60) % 60)) minutes and $(($diff % 60)) seconds elapsed."

echo "$(( ($diff / 3600) / 24 )) days, $(( ($diff / 3600) % 24 )) hours, $((($diff / 60) % 60)) minutes and $(($diff % 60)) seconds elapsed."

echo "$(( ( finnish - start ) )) seconds"
