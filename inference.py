import os
import sys
import torch

sys.path.append("/data/adhinart/freseg")

from freseg_inference import evaluation
import argparse
from model import RandLANet
from functools import partial

def load_model(output_dir, neighbors, decimation):
    d_in = 3
    num_classes = 2

    model = RandLANet(
        d_in,
        num_classes,
        num_neighbors=neighbors,
        decimation=decimation,
        device=torch.device("cuda"),
    )


    state_dict = torch.load(os.path.join(output_dir, "checkpoint_100.pth"))["model_state_dict"]
    model.load_state_dict(state_dict)
    
    return model

def model_inference(model, points):
    # 16, 2, 30000: [B, 2, N]
    scores = model(points)
    # B, N, 2
    scores = scores.permute(0, 2, 1)
    scores = scores.contiguous().view(-1, 2)
    pred_max = scores.max(1)[1]

    return pred_max

def main(args):
    exp_dir = f"./runs/{args.fold}_{args.path_length}_{args.npoint}_{args.frenet}"
    assert os.path.exists(exp_dir), f"Experiment {exp_dir} does not exist"

    evaluation(
        output_path=exp_dir,
        fold=args.fold,
        path_length=args.path_length,
        npoint=args.npoint,
        frenet=args.frenet,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        load_model=partial(load_model, neighbors=args.neighbors, decimation=args.decimation),
        model_inference=model_inference
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Testing")
    parser.add_argument(
        "--batch_size", type=int, default=16, help="batch Size during training"
    )
    parser.add_argument("--npoint", type=int, default=2048, help="point Number")
    parser.add_argument("--path_length", type=int, help="path length")
    parser.add_argument("--fold", type=int, help="fold")
    parser.add_argument("--num_workers", type=int, default=16, help="num workers")

    parser.add_argument(
        "--frenet", action="store_true", help="whether to use Frenet transformation"
    )
    parser.add_argument('--decimation', type=int, help='ratio the point cloud is divided by at each layer',
                        default=4)
    parser.add_argument('--neighbors', type=int, help='number of neighbors considered by k-NN',
                        default=16)

    args = parser.parse_args()

    main(args)
