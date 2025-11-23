
import argparse, os
from .datasets import get_cifar10, get_purchase_synth, split_train_val
from .sisa import sisa_train
from .utils import ensure_dir, set_seed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10","purchase_synth"], default="purchase_synth")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--S", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--log_csv", default="results/logs/baseline.csv")
    args = ap.parse_args()

    set_seed(args.seed); ensure_dir(os.path.dirname(args.log_csv))
    if args.dataset == "cifar10":
        train_full, _, _ = get_cifar10(args.data_root, train=True, download=True)
        testset, _, _   = get_cifar10(args.data_root, train=False, download=True)
        model_name = "cnn"
    else:
        train_full, _, _ = get_purchase_synth(args.data_root, train=True)
        testset, _, _    = get_purchase_synth(args.data_root, train=False)
        model_name = "mlp"

    trainset, valset = split_train_val(train_full, val_fraction=0.1, seed=args.seed)
    sisa_train(dataset=trainset, valset=valset, testset=testset, model_name=model_name,
               K=args.K, S=args.S, batch_size=args.batch_size, epochs_per_slice=args.epochs,
               lr=args.lr, seed=args.seed, log_path=args.log_csv, checkpoint_dir="results/ckpts")

if __name__ == "__main__":
    main()
