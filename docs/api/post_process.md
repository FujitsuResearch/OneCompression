# Post-Process

Post-quantization process classes for fine-tuning quantized models.

## Base Class

::: onecomp.post_process.PostQuantizationProcess
    options:
      show_source: false

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
