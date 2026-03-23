# PTPC-FP8

[vLLM PTPC-FP8 블로그](https://vllm.ai/blog/ptpc-fp8-rocm)를 PyTorch로 모사한 구현체.

## 핵심 아이디어

LLM activation의 outlier 문제를 이중 스케일로 해결:

| 대상 | 방식 | Scale |
|------|------|-------|
| Activation | Per-Token | `[L, 1]` |
| Weight | Per-Channel | `[W_out, 1]` |

두 scale 모두 합산 차원 `k`에 독립적이므로 GEMM 밖으로 분리 가능:

```
Y = (X_fp8 @ W_fp8^T) * (s_x @ s_w^T)
```

## 결과

```
Activation QDQ SQNR : 31.55 dB
Weight QDQ SQNR     : 31.56 dB
MatMul SQNR         : 28.53 dB
```
