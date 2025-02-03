import esm
import esm.modules  # pyright: ignore[reportMissingImports]
import esm2quinox
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
import torch


def test_transformer_layer(getkey):
    embed_size = 36
    hidden_size = 23
    num_heads = 3
    seq_len = 7
    assert (embed_size % num_heads) == 0

    torch_layer = esm.modules.TransformerLayer(
        embed_dim=embed_size,
        ffn_embed_dim=hidden_size,
        attention_heads=num_heads,
        add_bias_kv=False,
        use_esm1b_layer_norm=True,
        use_rotary_embeddings=True,
    )
    eqx_layer = esm2quinox.torch2eqx_transformer_layer(torch_layer)

    x = jr.normal(getkey(), (seq_len, embed_size))
    is_pad = jr.bernoulli(getkey(), 0.2, shape=(seq_len,))
    # Make sure have an example of each.
    is_pad = is_pad.at[1].set(True)
    is_pad = is_pad.at[2].set(False)
    torch_out, _ = torch_layer(
        torch.tensor(np.asarray(x))[:, None],
        self_attn_padding_mask=torch.tensor(np.asarray(is_pad))[None],
    )
    torch_out = torch_out.transpose(0, 1)  # (T, B) -> (B, T)
    torch_out = np.asarray(torch_out.detach())
    eqx_out = eqx_layer(x, is_pad=is_pad)[None]
    eqx_out = np.asarray(eqx_out)
    assert np.allclose(torch_out, eqx_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("token_dropout", (False, True))
def test_esm2(token_dropout, getkey):
    num_layers = 5
    embed_size = 36
    num_heads = 3
    assert (embed_size % num_heads) == 0

    with torch.random.fork_rng():
        torch.random.manual_seed(jr.randint(getkey(), shape=(), minval=0, maxval=2**30))
        torch_layer = esm.ESM2(  # pyright: ignore[reportAttributeAccessIssue]
            num_layers=num_layers, embed_dim=embed_size, attention_heads=num_heads
        )
        torch_layer.token_dropout = token_dropout
    eqx_layer = esm2quinox.torch2eqx_esm2(torch_layer)
    assert eqx_layer.token_dropout == token_dropout

    # Test every token -- and in particular MASK and PAD, for which special behaviour
    # applies.
    x = jnp.arange(33)

    with torch.inference_mode():
        torch_out = torch_layer(
            torch.tensor(np.asarray(x))[None], repr_layers=[num_layers]
        )
    torch_logits = np.asarray(torch_out["logits"])
    torch_hidden = np.asarray(torch_out["representations"][num_layers])
    eqx_out = eqx_layer(x)
    eqx_logits = np.asarray(eqx_out.logits[None])
    eqx_hidden = np.asarray(eqx_out.hidden[None])
    assert np.allclose(torch_hidden, eqx_hidden, rtol=1e-4, atol=1e-4)
    assert np.allclose(torch_logits, eqx_logits, rtol=1e-4, atol=1e-4)


def test_tokenise(getkey):
    proteins = ["SPIDERMAN", "FOO"]
    _, alphabet = esm.pretrained.esm2_t6_8M_UR50D()  # pyright: ignore[reportAttributeAccessIssue]
    converter = esm.data.BatchConverter(alphabet)  # pyright: ignore[reportAttributeAccessIssue]
    _, _, torch_tokenised = converter([("", protein) for protein in proteins])
    jax_tokenised = esm2quinox.tokenise(proteins)
    assert torch_tokenised.shape == jax_tokenised.shape
    assert np.all(np.asarray(torch_tokenised) == np.asarray(jax_tokenised))

    spid, foo = esm2quinox.tokenise(proteins, length=4, key=getkey())
    [true_spid] = esm2quinox.tokenise(["SPIDERMAN"])
    [true_foo] = esm2quinox.tokenise(["FOO"])
    assert spid.shape == (4,)
    assert true_spid.shape == (11,)
    assert foo.shape == (4,)
    assert true_foo.shape == (5,)
    assert np.all(foo == true_foo[1:]) or np.all(foo == true_foo[:-1])
    for i in range(8):
        if np.all(spid == true_spid[i : i + 4]):
            break
    else:
        assert False


@pytest.mark.parametrize("token_dropout", (False, True))
def test_call_on_string(token_dropout, getkey):
    model = esm2quinox.ESM2(
        num_layers=3,
        embed_size=16,
        num_heads=2,
        token_dropout=token_dropout,
        key=getkey(),
    )
    out = model("SPIDmmERMAN")
    [tokens] = esm2quinox.tokenise(["SPIDmmERMAN"])
    out2 = model(tokens)
    assert jnp.array_equal(out.hidden, out2.hidden)
    assert jnp.array_equal(out.logits, out2.logits)
