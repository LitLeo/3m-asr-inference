import numpy as np

def splice(feats, lctx, rctx, pad=True):
    """
    splice feature with context
    Args:
        - feats(numpy.float32): 2d feature to be spliced
        - lctx(integer): frames number of left context
        - rctx(integer): frames number of right context
        - pad(bool): only splice valid frames when false, otherwise
                     pad with zeros and splice each frame
    Returns:
        - spliced(numpy.float32): 2d spliced feature
    """
    frames, dim = feats.shape
    length = frames if pad else frames - lctx - rctx
    assert length > 0
    padding = feats
    if pad:
        l_pad = np.zeros((lctx, dim), dtype=np.float32)
        r_pad = np.zeros((rctx, dim), dtype=np.float32)
        padding = np.concatenate([l_pad, padding, r_pad], axis=0)
    spliced = np.zeros((length, (lctx + 1 + rctx) * dim), dtype=np.float32)
    for i in range(lctx + 1 + rctx):
        spliced[:, i*dim:(i+1)*dim] = padding[i:i+length, :]
    return spliced



def putThread(queue, generator, *gen_args):
    for item in generator(*gen_args):
        queue.put(item)
        if item is None:
            break

def getInputDim(args):
    return args.feats_dim * (args.lctx + 1 + args.rctx)
