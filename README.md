<h1 align="center">ESM2quinox</h1>
<h2 align="center">An implementation of [ESM2](https://github.com/facebookresearch/esm) in [Equinox+JAX](https://github.com/patric-kidger/equinox)</h2>

Only implements the representation trunk (not the structure-prediction head).

## Installation

```
pip install esm2quinox
```

## Public API

See their docstrings for details:
```
esm2quinox
    .ESM2
        .__init__(self, num_layers: int, embed_size: int, num_heads: int, token_dropout: bool, key: PRNGKeyArray)
        .__call__(self, tokens: Int[np.ndarray | jax.Array, " length"]) -> esm2quinox.ESM2Result

    .ESM2Result
        .hidden: Float[Array, "length embed_size"]
        .logits: Float[Array, "length alphabet_size"]

    .tokenise(proteins: list[str], length: None | int = None, key: None | PRNGKeyArray = None)

    .from_torch(torch_esm2: esm.ESM2) -> esm2quinox.ESM2
```

## Quick examples

Load an equivalent pretrained model from PyTorch:
```python
import esm  # pip install fair-esm==2.0.0
import esm2quinox

torch_model, _ = esm.pretrained.esm2_t6_8M_UR50D()
model = esm2quinox.from_torch(torch_model)
```

Create a randomly-initialised model:
```python
import esm2quinox
import jax.random as jr

key = jr.key(1337)
model = esm2quinox.ESM2(num_layers=3, embed_size=32, num_heads=2, token_dropout=False, key=key)
```

Forward pass (note the model operates on unbatched data):
```python
proteins = esm2quinox.tokenise(["SPIDERMAN", "FOO"])
out = jax.vmap(model)(proteins)
out.hidden  # hidden representation from last layer
out.logits  # logits for masked positions
```
