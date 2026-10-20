# Verification of the three existing references

This review retains all three works and uses formal publication metadata where
verified. It inspects papers and official documentation, without running their
software. Review date: 2026-10-19 (local time).

| Reference | Bibliographic correction or confirmation | Evidence used for comparison |
| --- | --- | --- |
| David Saxton, Edward Grefenstette, Felix Hill, Pushmeet Kohli, *Analysing Mathematical Reasoning Abilities of Neural Models* | ICLR 2019, rather than only an arXiv citation. | [Official record](https://openreview.net/forum?id=H1gR5iR5FX); [author PDF](https://arxiv.org/pdf/1904.01557), Sections 2.1-2.5 and Appendix B. Typed module composition, procedural school-level problems, interpolation/extrapolation and exact textual answers are already established. |
| Guillaume Lample, Francois Charton, *Deep Learning for Symbolic Mathematics* | ICLR 2020; 2019 is the preprint year. The bibliography preserves the accent in Charton's given name. | [Official record](https://openreview.net/forum?id=S1eZYeHFDS); [author PDF](https://arxiv.org/pdf/1912.01412), Sections 2-3, 4.3 and Appendix C/E. Generation uses mathematical identities; the paper also studies generalization across generation schemes. |
| Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy Liang, Quoc V. Le, Tengyu Ma, Adams Wei Yu, *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining* | NeurIPS 2023, volume 36. Existing author order and family/given-name fields confirmed; the citation now identifies the formal proceedings. | [Formal record](https://proceedings.neurips.cc/paper_files/paper/2023/hash/dcba6be91359358c2355cd920da3fcbd-Abstract-Conference.html); [paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/dcba6be91359358c2355cd920da3fcbd-Paper-Conference.pdf), Section 2/Algorithm 1, Sections 3.1 and 4, Appendix C. Token-normalized excess loss and a small reference/proxy pair determine weights for larger-model training. Downstream tasks do not determine those weights. |

Direct OpenReview PDF requests encountered a browser challenge. The accessible
author PDFs identify the ICLR publication on their title pages; official
repository documentation also identifies Symbolic Mathematics as ICLR 2020.
No access restriction was bypassed. External metadata checks do not establish
that the released code exactly reproduces a paper.

## Official code and documentation inspected

- **Mathematics Dataset**, revision
  [`427f45075f84b8b9774950196ad63867ca20ffb3`](https://github.com/google-deepmind/mathematics_dataset/tree/427f45075f84b8b9774950196ad63867ca20ffb3):
  [README](https://github.com/google-deepmind/mathematics_dataset/blob/427f45075f84b8b9774950196ad63867ca20ffb3/README.md),
  [`generate.py`](https://github.com/google-deepmind/mathematics_dataset/blob/427f45075f84b8b9774950196ad63867ca20ffb3/mathematics_dataset/generate.py),
  and [`util/composition.py`](https://github.com/google-deepmind/mathematics_dataset/blob/427f45075f84b8b9774950196ad63867ca20ffb3/mathematics_dataset/util/composition.py).
  Module filters, entropy settings, sample counts and type-compatible composition
  are explicit controls. Composition constructs a question from subproblems;
  it should not be conflated with a registry of jointly sufficient answers
  for separate tasks on the same permutation.
- **SymbolicMathematics**, revision
  [`4596d070e1a9a1c2239c923d7d68fda577c8c007`](https://github.com/facebookresearch/SymbolicMathematics/tree/4596d070e1a9a1c2239c923d7d68fda577c8c007):
  [README sections Data generation, Evaluation, and FAQ](https://github.com/facebookresearch/SymbolicMathematics/blob/4596d070e1a9a1c2239c923d7d68fda577c8c007/README.md).
  Generator options bound operators, integer magnitudes and expression lengths.
  Symbolic/numerical solution checks, deduplication, splits, mixtures, and
  generation-scheme cross-evaluation are documented. The README explicitly
  distinguishes exact-string validation from final solution verification and
  warns that its downloadable datasets/models differ slightly from the paper.
- **DoReMi**, revision
  [`7cde52d1848737aa967ecbdb9e643cf334de160d`](https://github.com/sangmichaelxie/doremi/tree/7cde52d1848737aa967ecbdb9e643cf334de160d):
  [README sections Getting started, Running DoReMi on your own dataset,
  and Tips and details](https://github.com/sangmichaelxie/doremi/blob/7cde52d1848737aa967ecbdb9e643cf334de160d/README.md).
  Domain-weight configuration, resumable weighted sampling, a training script
  and downstream evaluation are public. The public PyTorch implementation
  documents differences from the paper's JAX implementation, including
  architecture and tokenization; token shares depend on the tokenizer.

For these works, a systematic single/joint/transformed-query relation registry
with separate proof and finite-check status remains **unverified in the inspected
sources**, rather than asserted absent. Their established use of mathematical
structure or budgeted mixture optimization must still be credited.
