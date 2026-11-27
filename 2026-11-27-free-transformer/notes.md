# Daily Experiment: Free Transformer (2026-11-27)

## Source Tweet
https://x.com/francoisfleuret/status/1987083801135055065

**Claim**: Slides for talk on Free Transformer — an extension of decoder-only Transformers that conditions generation on unsupervised learned random latent variables (via conditional VAE) with only ~3-4% overhead, improving reasoning/code/math tasks.

Paper: https://arxiv.org/abs/2510.17558  
PDF: https://arxiv.org/pdf/2510.17558.pdf  
Slides: https://fleuret.org/public/slides-talk-free-transformer.pdf

## Architecture Summary
- Inject discrete one-hot latent Z (dim 2^H, typically H=16 → 65k states) into the middle of the decoder stack.
- During training/prefill: lightweight encoder (one non-causal Transformer block + linear) on the first half of layers predicts/samples Z consistent with the sequence (CVAE).
- During pure generation: sample Z uniformly from the discrete space.
- Shares first L/2 blocks between encoder and decoder → minimal compute overhead.
- Loss: standard cross-entropy + KL(Q(Z|S) || P(Z)).

Key motivation (from paper): pure AR models must infer high-level decisions (sentiment, plan, structure) post-hoc from noisy tokens; explicit latent allows the model to "decide once" and stick to it.  
Example: movie review generator can commit to positive/negative up front instead of gradually inferring.

## Toy Experiment
Implemented a minimal prototype comparing:
1. Standard causal Transformer (4 layers)
2. Free-style: first half layers → encoder samples global binary Z → inject into residual → second half layers

**Task**: Binary sequences of length 32 where a latent mode Z ~ Bern(0.5) sets the majority bit, each token flipped with ε=0.1. Pure AR must estimate posterior from previous noisy tokens; Free can use explicit Z.

Code: `toy_free_transformer.py`

### Results (15 epochs, CPU, small models)
| Model          | Final Val NLL | Final Val Acc |
|----------------|---------------|---------------|
| Standard AR    | 0.3274        | 0.8966        |
| Free-style     | 0.3264        | 0.8979        |

Slight edge to Free-style on both NLL and accuracy. Generations from both look consistent with the noisy majority-bit structure (as expected for this easy task).

Plot: `free_transformer_toy_results.png`

### Limitations of this toy
- Global (sequence-level) Z instead of per-token.
- No explicit KL annealing / free bits.
- Very small capacity; task is solvable by AR with running counts.
- No large-scale validation (paper trains 1.5B/8B from scratch).

## Paper Results (from arXiv:2510.17558 and secondary reports)
- 1.5B models: up to +44-55% on HumanEval+ (code), solid gains on GSM8K.
- 8B / 1T tokens: +11% code generation, smaller but positive gains on knowledge/math.
- Overhead: ~3-4% compute/memory.
Sources: https://arxiv.org/abs/2510.17558 , https://the-decoder.com/metas-free-transformer-introduces-a-new-approach-to-llm-decision-making/

## Improvement Insights
1. **Per-token vs global Z**: Paper uses per-position Z; toy used global. Per-token allows finer structure (e.g., position of a pattern) but requires careful information bottleneck.
2. **KL schedule / free bits**: Critical to prevent posterior collapse or encoder cheating. Paper uses thresholding.
3. **Injection depth**: Paper notes early injection weakens encoder capacity, late weakens decoder's ability to use Z. Ablate systematically.
4. **Combine with CoT / RL reasoning**: Latent decisions in activation space + explicit token CoT could be complementary (mentioned in OpenReview discussion).
5. **Continuous or hierarchical latents**: Discrete one-hot is simple but limited; Gumbel-softmax or hierarchical VAE might capture richer plans.
6. **Open implementations**: Unofficial PyTorch in https://github.com/udapy/free-transformer and lucidrains/x-transformers (`FreeTransformer`).

## Visuals
- `/workspace/artifacts/free_transformer_toy_results.png` — val NLL and accuracy curves. Attach to show the small but consistent edge on the synthetic latent-conditioned task.

## Next Steps / Collaboration
- Scale the toy to a real small language modeling setup (e.g., character-level or TinyStories) with proper KL.
- Port the exact architecture from the paper (Binary Mapper, mid-layer injection with shared blocks).
- Benchmark against baseline on a reasoning synthetic (e.g., arithmetic with carry that requires early commitment).

All facts above are sourced from the linked paper, slides, or the experiment code run today.
