import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import math
import torch
import torch.nn as nn

from transformer.transformer import (
    # utilities
    stable_softmax,
    cross_entropy_logits,
    char_tokenize,
    generate_causal_mask,
    generate_padding_mask,
    positional_encoding,
    # core modules
    FeedForward,
    MultiHeadAttention,
    MultiHeadCrossAttention,
    PreLNResidual,
    TransformerBlock,
    Encoder,
    Decoder,
    # embeddings / wrappers
    TokenEmbedding,
    Seq2SeqTransformer,
    # training helpers
    build_optimizer_and_scheduler,
    train_one_epoch,
    set_seed,
)


# -----------------------------
# Utilities
# -----------------------------

def test_stable_softmax_properties():
    x = torch.tensor([[1.0, 2.0, 3.0]])
    p = stable_softmax(x, dim=-1)
    assert p.shape == x.shape
    assert torch.isfinite(p).all()
    # normalized
    assert torch.allclose(p.sum(dim=-1), torch.ones(1))
    # monotonic order preserved for this vector
    assert torch.argmax(p, dim=-1).item() == 2


def test_cross_entropy_logits_2d_and_3d():
    torch.manual_seed(0)
    # 2D case
    logits2 = torch.randn(5, 7)
    targets2 = torch.randint(0, 7, (5,))
    loss2 = cross_entropy_logits(logits2, targets2)
    assert loss2.dim() == 0 and torch.isfinite(loss2)

    # 3D case
    B, T, V = 2, 3, 11
    logits3 = torch.randn(B, T, V)
    targets3 = torch.randint(0, V, (B, T))
    loss3 = cross_entropy_logits(logits3, targets3)
    assert loss3.dim() == 0 and torch.isfinite(loss3)


def test_char_tokenize():
    assert char_tokenize("hey") == ["h", "e", "y"]


def test_generate_padding_mask_bool_and_additive():
    key_lengths = [2, 3]
    T = 4
    m_bool = generate_padding_mask(key_lengths, T=T, device='cpu', mode='bool')
    assert m_bool.shape == (2, 1, 1, 4)
    expected = torch.tensor([
        [[[False, False, True,  True]]],
        [[[False, False, False, True]]],
    ], dtype=torch.bool)
    assert torch.equal(m_bool, expected)

    mv = -1e9
    m_add = generate_padding_mask(key_lengths, T=T, device='cpu', mode='additive', masked_value=mv)
    assert m_add.shape == (2, 1, 1, 4)
    # places matching True in bool mask should be mv here
    assert torch.allclose(m_add[0,0,0], torch.tensor([0.0, 0.0, mv, mv]))


def test_generate_causal_mask():
    L = 5
    m = generate_causal_mask(L, device='cpu')
    assert m.shape == (1, 1, L, L)
    tri = torch.ones((L, L), dtype=torch.bool).triu(1)
    assert torch.equal(m[0, 0], tri)


def test_positional_encoding_properties():
    L, D = 6, 8
    PE = positional_encoding(L, D)
    assert PE.shape == (L, D)
    assert torch.isfinite(PE).all()
    assert torch.allclose(PE[0, 0::2], torch.zeros(D // 2))
    assert torch.allclose(PE[0, 1::2], torch.ones(D // 2))
    s = PE[:, 0::2]
    c = PE[:, 1::2]
    assert torch.allclose(s.pow(2) + c.pow(2), torch.ones_like(s), atol=1e-6)


# -----------------------------
# Core modules
# -----------------------------

def test_feedforward_shapes_and_grad():
    torch.manual_seed(0)
    B, T, D = 2, 3, 16
    ff = FeedForward(D, expansion=4, dropout=0.0)
    x = torch.randn(B, T, D, requires_grad=True)
    y = ff(x)
    assert y.shape == (B, T, D)
    y.sum().backward()
    assert x.grad is not None


def test_multihead_attention_shapes_and_mask():
    torch.manual_seed(0)
    B, L, D, H = 2, 4, 8, 2
    mha = MultiHeadAttention(D, H)
    x = torch.randn(B, L, D)
    # no mask
    out = mha(x)
    assert out.shape == (B, L, D)
    # causal mask shape broadcast
    mask = torch.ones((1,1,L,L), dtype=torch.bool).triu(1)
    out2 = mha(x, attn_mask=mask)
    assert out2.shape == (B, L, D)


def test_multihead_cross_attention_shapes():
    torch.manual_seed(0)
    B, Tq, Tk, D, H = 2, 5, 7, 12, 3
    xq = torch.randn(B, Tq, D)
    kv = torch.randn(B, Tk, D)
    m = MultiHeadCrossAttention(D, H)
    # pad mask over encoder keys
    pad_mask = generate_padding_mask([Tk]*B, T=Tk, device='cpu', mode='bool')
    out = m(xq, kv, attn_mask=pad_mask)
    assert out.shape == (B, Tq, D)


def test_preln_residual_wraps_module():
    torch.manual_seed(0)
    B, T, D = 2, 3, 8
    sub = nn.Linear(D, D)
    wrap = PreLNResidual(D, sub, dropout=0.0)
    x = torch.randn(B, T, D)
    y = wrap(x)
    assert y.shape == (B, T, D)


def test_transformer_block_composition():
    torch.manual_seed(0)
    B, L, D, H = 2, 6, 16, 4
    blk = TransformerBlock(D, H)
    x = torch.randn(B, L, D)
    mask = torch.ones((1,1,L,L), dtype=torch.bool).triu(1)
    y = blk(x, attn_mask=mask)
    assert y.shape == (B, L, D)


# -----------------------------
# Encoder / Decoder
# -----------------------------

def test_encoder_forward_shapes():
    torch.manual_seed(0)
    B, SrcT, D, H, L = 2, 5, 16, 4, 2
    enc = Encoder(D, H, L)
    src = torch.randn(B, SrcT, D)
    out = enc(src, src_lengths=[SrcT]*B)
    assert out.shape == (B, SrcT, D)


def test_decoder_forward_shapes_with_masks():
    torch.manual_seed(0)
    B, SrcT, TgtT, D, H, L = 2, 4, 6, 16, 4, 2
    dec = Decoder(D, H, L)
    # decoder takes embedded tgt and encoder memory (src already encoded), but here we can use rand tensors
    mem = torch.randn(B, SrcT, D)
    tgt = torch.randn(B, TgtT, D)
    y = dec(tgt, mem, tgt_lengths=[TgtT]*B, src_lengths=[SrcT]*B)
    assert y.shape == (B, TgtT, D)


# -----------------------------
# Embedding, wrappers, and end-to-end
# -----------------------------

def test_token_embedding_and_tied_weights():
    V, D = 20, 12
    emb = TokenEmbedding(V, D)
    idx = torch.tensor([[0,1,2]])
    out = emb(idx)
    assert out.shape == (1, 3, D)

    model = Seq2SeqTransformer(vocab_size=V, d_model=D, num_heads=3, num_layers=1)
    # tied: projection weight object is the same as embedding weight tensor
    assert model.output_proj.weight.data_ptr() == model.tgt_embed.weight.weight.data_ptr()


def test_seq2seq_transformer_decoder_only_forward_and_loss():
    torch.manual_seed(42)
    V, D, H, L = 32, 16, 4, 2
    B, T = 3, 7

    model = Seq2SeqTransformer(vocab_size=V, d_model=D, num_heads=H, num_layers=L)

    # Decoder-only: src_ids=None; targets are next-token labels
    tgt_in = torch.randint(0, V, (B, T))
    tgt_out = torch.roll(tgt_in, shifts=-1, dims=1)

    logits = model(src_ids=None, tgt_in_ids=tgt_in)
    assert logits.shape == (B, T, V)

    loss = model.loss(logits, tgt_out, label_smoothing=0.1)
    assert loss.item() == loss.item()  # finite
    loss.backward()


# -----------------------------
# Training helpers
# -----------------------------

def test_set_seed_reproducibility():
    set_seed(123)
    a = torch.rand(3)
    set_seed(123)
    b = torch.rand(3)
    assert torch.allclose(a, b)


def test_build_optimizer_and_scheduler_warmup_then_decay():
    # tiny model
    model = nn.Linear(4, 4)
    opt, sched = build_optimizer_and_scheduler(model, lr=1e-3, warmup_steps=3, total_steps=10)
    lrs = []
    for step in range(10):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]['lr'])
    # LR should rise for first ~3 steps then begin to decay (cosine)
    assert lrs[1] > lrs[0]
    assert lrs[2] >= lrs[1]
    assert lrs[-1] <= lrs[2]


def test_train_one_epoch_decoder_only_smoke():
    class TinyData:
        def __iter__(self):
            for _ in range(20):
                B, T, V = 4, 16, 32
                x = torch.randint(0, V, (B, T))
                y = torch.roll(x, shifts=-1, dims=1)
                yield {
                    'src_ids': None,
                    'tgt_in_ids': x,
                    'tgt_out_ids': y,
                    'src_lengths': None,
                    'tgt_lengths': [T]*B,
                }

    torch.manual_seed(0)
    V, D, H, L = 32, 16, 4, 2
    model = Seq2SeqTransformer(vocab_size=V, d_model=D, num_heads=H, num_layers=L)
    opt, sched = build_optimizer_and_scheduler(model, lr=5e-4, warmup_steps=1, total_steps=50)

    avg = train_one_epoch(model, TinyData(), opt, sched, device=torch.device('cpu'),
                          label_smoothing=0.1, grad_clip=1.0, log_every=100)
    assert math.isfinite(avg)