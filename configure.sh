qsub -I -q gpu -l select=1:ncpus=1:ngpus=1:gpu_mem=15gb:mem=10gb -l walltime=24:0:0
cd /auto/brno2/home/hajkoj/M3RT
source env/bin/activate
module add python/python-3.10.4-intel-19.0.4-sc7snnf
export PYTHONPATH=/auto/brno2/home/hajkoj/M3RT/src/retrievers