"""Functions for generating summaries of training
progress in the middle of the training process"""
# from torch.utils.tensorboard import SummaryWriter


def callback(n_iter, cache_df, name):
    print(cache_df.mean().to_dict())


# writer = SummaryWriter()


def tensorboard_callback(n_iter, cache_df, name):
    # for k, v in cache_df.mean().to_dict().items():
    #     writer.add_scalar(f"{k}/{name}", v, n_iter)
    pass  # No code uses tensorboard any more


logger = []


def log_callback(n_iter, cache_df, name):
    d = cache_df.mean().to_dict()
    d["n_iter"] = n_iter
    logger.append(d)
