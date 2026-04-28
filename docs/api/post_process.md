# Post-Process

Post-quantization process classes for improving quantized model accuracy.

## Base Class

::: onecomp.post_process.PostQuantizationProcess
    options:
      show_source: false

## Block-wise PTQ

::: onecomp.post_process.BlockWisePTQ
    options:
      show_source: false
      members:
        - run

## LoRA SFT

::: onecomp.post_process.PostProcessLoraSFT
    options:
      show_source: false
      members:
        - run

## Convenience Variants

`PostProcessLoraTeacherSFT` and `PostProcessLoraTeacherOnlySFT` are pre-configured
variants of `PostProcessLoraSFT` with different default loss weights:

::: onecomp.post_process.PostProcessLoraTeacherSFT
    options:
      show_source: false
      show_bases: false
      members: false

::: onecomp.post_process.PostProcessLoraTeacherOnlySFT
    options:
      show_source: false
      show_bases: false
      members: false
