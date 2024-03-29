import torch

from lib.functions.activations import log_softmax


class Path:
    def __init__(self, seq: list, logp: list, state=None, ended=False):
        self.seq = seq
        self.logp = logp
        self.state = state
        self.ended = ended

    def __repr__(self):
        return f"\n Path(seq={self.seq}, ended={self.ended})"


def greedy_search(decoder, context, start_token: int, end_token: int, max_steps=10):
    assert type(start_token) is int and type(end_token) is int, f'start_token/end_token should be an integer but got {start_token}/{end_token} (note there is no batch support)'
    seq = []
    token = torch.tensor(start_token)
    state = context
    for t in range(max_steps):
        z, state = decoder.forward(token.view(1, 1), state)
        token = z.squeeze().argmax()
        seq.append(token.item())
        if token == end_token:
            break
    return seq


def beam_search(decoder, context, start_token: int, end_token: int, max_steps=10, k=2, alpha=.75):
    assert type(start_token) is int and type(end_token) is int, f'start_token/end_token should be an integer but got {start_token}/{end_token} (note there is no batch support)'

    # Define the scoring function
    score_fn = lambda p: sum(p.logp) / len(p.seq) ** alpha

    # Initialize with empty path
    paths = [Path(seq=[], logp=[], state=context, ended=False)]
    for t in range(max_steps):
        candidates = []

        # Expand each path (in total k times)
        for path in paths:
            if path.ended:
                candidates.append(path)
                continue

            # Decode and select top k tokens
            last_token = path.seq[-1] if len(path.seq) else start_token
            z, state = decoder.forward(torch.tensor(last_token).view(1, 1), path.state)
            k_logp, k_tokens = log_softmax(z).squeeze().topk(k)

            # Collect top k candidate sequences
            for logp, token in zip(k_logp, k_tokens):
                seq = path.seq + [token.item()]
                logp = path.logp + [logp.item()]
                candidates.append(Path(seq, logp, state, ended=token.item() == end_token))

        # Score and prune
        candidates.sort(key=score_fn, reverse=True)
        paths = candidates[:k]
        # assert k <= len(candidates) <= k**2
        # print(t, paths)
        if all(path.ended for path in paths):
            break  # all selected paths are completed

    best = max(paths, key=score_fn)
    return best.seq, score_fn(best)

