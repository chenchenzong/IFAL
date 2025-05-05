

## Run 

```
python main.py --gpu 0 --seed 1 --al_method ifal --model cnn4conv --dataset cifar10 --partition dir_balance --dd_beta 0.1 --num_users 10 --frac 1.0 --num_classes 10 --rounds 100 --local_ep 5 --reset random --query_ratio 0.05

python main.py --gpu 0 --seed 1 --al_method ifal --model cnn4conv --dataset cifar100 --partition dir_balance --dd_beta 0.1 --num_users 10 --frac 1.0 --num_classes 100 --rounds 100 --local_ep 5 --reset random --query_ratio 0.05

python main.py --gpu 0 --seed 1 --al_method ifal --model cnn4conv --dataset tinyimagenet --partition dir_balance --dd_beta 0.1 --num_users 10 --frac 1.0 --num_classes 200 --rounds 100 --local_ep 5 --reset random --query_ratio 0.05

```

