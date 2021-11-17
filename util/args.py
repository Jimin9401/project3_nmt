import argparse
import os
class NMTArgument:
    def __init__(self):
        data = {}
        parser = self.get_args()
        args = parser.parse_args()

        data.update(vars(args))
        self.data = data
        self.set_savename()
        self.__dict__ = data

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--root", type=str, default="data")
        parser.add_argument("--n_epoch", default=10, type=int)
        parser.add_argument("--seed", default=777, type=int)
        parser.add_argument("--per_gpu_train_batch_size", default=64, type=int)
        parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int)
        parser.add_argument("--gradient_accumulation_step", default=1, type=int)
        parser.add_argument("--seq_len", default=512, type=int)
        parser.add_argument("--warmup_step", default=0, type=int)
        parser.add_argument("--decay_step", default=20000, type=int)
        parser.add_argument("--clip_norm", default=0.25, type=float)
        parser.add_argument("--replc", default=0.25, type=float)

        parser.add_argument("--lr", default=1e-5, type=float)

        parser.add_argument("--weight_decay", default=0.0, type=float)
        parser.add_argument("--do_train", action="store_true")
        parser.add_argument("--do_eval", action="store_true")
        parser.add_argument("--evaluate_during_training", action="store_true")
        parser.add_argument("--do_test", action="store_true")
        parser.add_argument("--checkpoint_dir", default="checkpoints", type=str)
        parser.add_argument("--max_train_steps", default=None, type=int)
        parser.add_argument("--lr_scheduler_type", default="linear", help="The scheduler type to use.",
                            choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant",
                                     "constant_with_warmup"])

        parser.add_argument("--num_warmup_steps", type=int, default=0,
                            help="Number of steps for the warmup in the lr scheduler.")
        parser.add_argument("--replace_vocab", action="store_true")

        return parser

    def set_savename(self):

        self.data["savename"] = os.path.join(self.data["checkpoint_dir"], f"vanila")

