"""TF_GLM_SELECT_SPLIT: position-split selection equals the replicated selection, token for token, on every rank.

One GPU plays the four ranks: each rank's call first runs with a capturing comm (its candidates recorded), then again
with a replaying comm that hands every rank's recorded candidates back, as the real all-gather would."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)


class _Comm:
    def __init__(self, store: dict, rank: int, world: int, replay: bool):
        self.store, self.rank, self.world, self.replay = store, rank, world, replay

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        if self.replay:
            recv.copy_(torch.cat([self.store[r] for r in range(self.world)]))
        else:
            self.store[self.rank] = send.clone()
            recv.zero_()


@pytest.mark.parametrize("fmt", ["bf16", "fp8"])
@pytest.mark.parametrize("R,pos,prompt", [(1, 30000, False), (4, 9000, False), (8, 2040, False), (40, 14000, True),
                                          (300, 5000, True), (40, 2030, True), (130, 1500, True)])
def test_split_selection_equals_replicated(fmt, R, pos, prompt, monkeypatch):
    from tensorfold.families.glm_moe_dsa.cuda import kv8, select

    g = torch.Generator().manual_seed(R * 31 + pos)
    HI, D, topk, world = 32, 128, 2048, 4
    n = pos + R + 64
    keys = (torch.randn((n, D), generator=g) * (torch.rand((D,), generator=g) + 0.2)).to(torch.bfloat16).cuda()
    if fmt == "fp8":
        plane = kv8.Kv8(n, D, "cuda")
        kv8.write(keys, plane, torch.zeros((1,), dtype=torch.int32, device="cuda"))
        keys = plane
    qi = torch.randn((R, HI * D), generator=g).to(torch.bfloat16).cuda()
    wts = torch.randn((R, HI), generator=g).to(torch.bfloat16).cuda()
    pos_dev = torch.tensor([pos], dtype=torch.int32, device="cuda")
    host = pos if prompt else None
    bucket = None if prompt else select.sparse_bucket(pos, R)
    span = True if prompt else None

    def run(**kw):
        tokens = torch.empty((R, topk + 1), dtype=torch.int32, device="cuda")
        counts = torch.empty((R,), dtype=torch.int32, device="cuda")
        select.select_tokens(qi, wts, keys, host, R, topk, pos_dev, tokens=tokens, counts=counts, bucket=bucket,
                             span=span, **kw)
        torch.cuda.synchronize()
        return tokens, counts

    want_t, want_c = run()
    monkeypatch.setattr(select, "SPLIT", True)
    monkeypatch.setattr(select, "SPLIT_DECODE", 0)       # every decode window takes the split path here
    store: dict = {}
    for r in range(world):
        run(comm=_Comm(store, r, world, False), rank=r, world=world)
    for r in range(world):
        got_t, got_c = run(comm=_Comm(store, r, world, True), rank=r, world=world)
        assert torch.equal(got_c, want_c), r
        sparse = want_c > 0
        assert torch.equal(got_t[sparse], want_t[sparse]), (r, int(sparse.sum()))
