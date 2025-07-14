mkdir -p logs

python train.py --run without_desc --layers 2 > logs/log_without_desc_2layers.txt
python train.py --run without_graph --layers 2 > logs/log_without_graph_2layers.txt

python train.py --run att_baseline > logs/att_baseline.txt
python train.py --run geom_01 --layers 1 > logs/log_geom_01_l1.txt
python train.py --run geom_01 --layers 3 > logs/log_geom_01_l3.txt

python train.py --run baseline > logs/log_baseline_small_lr.txt

python train.py --run geom_01 --layers 2 > logs/log_geom_01.txt
python train.py --run geom_02 --layers 2 > logs/log_geom_02.txt
python train.py --run geom_03 --layers 2 > logs/log_geom_03.txt

python train.py --run pool > logs/log_pool.txt
