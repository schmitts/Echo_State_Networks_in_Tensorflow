#!/bin/bash
#SBATCH -p medium
#SBATCH -o /usr/users/jmeyer3/ownCloud/hpc_outfiles/outfile-%J #./../Results/hpc-outfiles/outfile-%J
#SBATCH -N 1
#SBATCH -n 24 #multiples of 24
#SBATCH -t 4-0:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jmeyer3@gwdg.de
#SBATCH --mem=300G
#SBATCH --qos=long
#SBATCH -C scratch

start=$(date +%s)
echo $date

script=grid_search.py 

cmd="python3 $script"
cmd="$cmd --min_exp 1"
cmd="$cmd --max_exp 4"
cmd="$cmd --min_coeff 0"
cmd="$cmd --max_coeff 1"
cmd="$cmd --periodicity 4"

echo $cmd
eval $cmd

finnish=$(date +%s)

diff=$(( ( finnish - start ) )) #$SECONDS

echo "$(($diff / 3600)) hours, $((($diff / 60) % 60)) minutes and $(($diff % 60)) seconds elapsed."

echo "$(( ($diff / 3600) / 24 )) days, $(( ($diff / 3600) % 24 )) hours, $((($diff / 60) % 60)) minutes and $(($diff % 60)) seconds elapsed."

echo "$(( ( finnish - start ) )) seconds"

