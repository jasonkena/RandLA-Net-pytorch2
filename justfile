default:
    just --list


# pathlength 10000 npoints 4096
train cuda fold pathlength npoints:
    CUDA_VISIBLE_DEVICES={{cuda}} python train.py --fold {{fold}} --path_length {{pathlength}} --npoint {{npoints}}

train_frenet cuda fold pathlength npoints:
    CUDA_VISIBLE_DEVICES={{cuda}} python train.py --fold {{fold}} --path_length {{pathlength}} --npoint {{npoints}} --frenet

# pathlength 10000 npoints 4096
evaluate cuda fold pathlength npoints:
    CUDA_VISIBLE_DEVICES={{cuda}} python inference.py --fold {{fold}} --path_length {{pathlength}} --npoint {{npoints}}

evaluate_frenet cuda fold pathlength npoints:
    CUDA_VISIBLE_DEVICES={{cuda}} python inference.py --fold {{fold}} --path_length {{pathlength}} --npoint {{npoints}} --frenet

