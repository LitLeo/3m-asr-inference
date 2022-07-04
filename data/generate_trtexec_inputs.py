import numpy as np
import sys

def main(B, S, D):
    dim_str = f"{B}x{S}x{D}"
    print("Generating data, dim=" + dim_str)
    feat = np.random.rand(B, S, D).astype(np.float32)
    feat_len = np.ones((B), dtype=np.int32) * S
    # print(feat)
    # print(feat.dtype)
    # print(feat_len)
    # print(feat_len.dtype)
    feat.tofile("feat." + dim_str + ".bin")
    feat_len.tofile("feat_len." + dim_str + ".bin")

if __name__ == '__main__':
    argc = len(sys.argv)
    if argc != 1 and argc != 4:
        print("Usage: python generate_trtexec_inputs.py B S D")
        print("default is python generate_trtexec_inputs.py 1 512 40")
        exit(0)

    if argc == 1:
        B = 1
        S = 512
        D = 40

    if argc == 4:
        B = int(sys.argv[1])
        S = int(sys.argv[2])
        D = int(sys.argv[3])

    main(B, S, D)
