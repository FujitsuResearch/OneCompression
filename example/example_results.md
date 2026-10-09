# Example Run Summary of OneComp v1.4.0

- Run date: 2026-10-09

## No. 0: `example_gptq.py`

- Example: GPTQ 3-bit
- Original model perplexity: 7.770143034789174
- Quantized model perplexity: 53.38688319438883
- Note: With 3-bit per-channel quantization, substantial quality degradation is expected.

## No. 1: `example_qep_gptq.py`

- Example: GPTQ 3-bit + QEP
- Original model perplexity: 7.770143034789174
- Quantized model perplexity: 15.363539314138155
- Note: QEP significantly mitigates the quality drop even under 3-bit per-channel quantization.

## No. 2: `example_jointq.py`

- Example: JointQ 4-bit
- Original model perplexity: 7.770143034789174
- Dequantized model perplexity: 8.092134952089124

## No. 3: `example_autobit.py`

- Example: AutoBitQuantizer
- Original model perplexity: 7.770143034789174
- Quantized model perplexity: 8.814287247154967

## No. 4: `example_auto_run.py`

- Example: Runner.auto_run
- Result: this example runs Runner.auto_run twice.
- Run 1: automatic VRAM estimation on NVIDIA B200 (target wbits = 1171.79, effectively assigned as GPTQ 8-bit / group size 128 for all 154 layers)
- Run 1 quantized model perplexity: 7.776067678562315
- Run 1 accuracy: ARC-Easy 0.6039562289562289, ARC-Challenge 0.28071672354948807, PIQA 0.7312295973884657, Winogrande 0.5895816890292028
- Run 2: user-specified VRAM = 1.00 GB (target wbits = 5.12)
- Run 2 quantized model perplexity: 8.115127657396819
- Run 2 accuracy: ARC-Easy 0.5883838383838383, ARC-Challenge 0.28924914675767915, PIQA 0.7295973884657236, Winogrande 0.584846093133386

## No. 5: `example_save_load.py`

- Example: Save -> Load -> Generate
- Result: quantized model was saved to ./tinyllama_gptq4 and reloaded on cuda:0.
- Prompt: Fujitsu is
- Generated text: Fujitsu is a Japanese multinational information technology company headquartered in Tokyo, Japan. Fujitsu is the world's third largest IT services company,

## No. 6: `example_custom_calibration.py`

- Example: Custom calibration
- Result: the example compared default C4 calibration against a custom Python-code calibration set.
- Observation: the custom calibration output was consistently more code-like for programming prompts such as def fibonacci(n): and def binary_search(arr, target):.
- Observation: both calibration modes still produced weak factual generation for the prompt The capital of France is.

## No. 7: `pre_process/example_llama_preprocess_rtn.py`

- Example: Rotation + RTN quantization
- Original model perplexity: 7.769899079011024
- Dequantized model perplexity: 11.592024097571747

## No. 8: `pre_process/example_preprocess_save_load.py`

- Example: Rotation + GPTQ -> Save -> Load
- Original model PPL: 7.77
- Rotated model PPL: 7.77
- Quantized model PPL: 8.83
- Loaded model PPL after save/load: 8.11
- Note: the reason why PPL improves after save/load is still under investigation.

## No. 9: `post_process/example_lora_sft.py`

- Example: LoRA SFT (WikiText-2)
- Original model PPL: 7.7701
- Quantized + LoRA SFT model PPL: 7.9461
- Result: the LoRA-applied GPTQ model was saved to ./tinyllama_gptq4_lora and reloaded on cuda:0.
- Prompt: Fujitsu is
- Generated text: Fujitsu is a Japanese multinational information technology company headquartered in Tokyo, Japan. It was founded on 1 April 1958 as the Fuji Electric Locomotive Works by Takeshi Furuta and Katsumi Toyama . The company's name is derived from the kanji characters for

## No. 10: `post_process/example_lora_sft_knowledge.py`

- Example: LoRA SFT (Knowledge injection)
- Result: the example compares generation before and after LoRA SFT with OneCompression knowledge data.
- Observation: before LoRA SFT, the model incorrectly described OneCompression as a generic file-compression method.
- Observation: after LoRA SFT, the model describes OneCompression as Fujitsu's open-source LLM quantization toolkit and mentions GPTQ, DBF, JointQ, RTN, and the Runner API.
- Result: the LoRA-applied model was saved to ./tinyllama_gptq4_lora_knowledge and reloaded on cuda:0.

## No. 11: `post_process/example_blockwise_ptq.py`

- Example: Block-wise PTQ
- Original model PPL: 7.7701
- Quantized + BlockWisePTQ PPL: 8.3910
- Result: the packed BlockWisePTQ checkpoint was saved to ./tinyllama-gptq-blockwise-packed.

## No. 12: `example_lpcd_gptq.py`

- Example: GPTQ 3-bit + QEP + LPCD
- Original model perplexity: 7.770143034789174
- Quantized model perplexity: 10.21667572581292
- Note: This example uses groupsize=128, so its result should not be directly compared with the per-channel GPTQ 3-bit examples above.

## No. 13: `post_process/example_global_ptq.py`

- Example: Global PTQ (GPTQ + KL distillation)
- Original PPL: 7.7701
- Quantized + Global PTQ PPL: 8.2438
- Result: the model was saved to ./tinyllama-gptq-globalptq.

## No. 14: `post_process/example_global_ptq_dbf.py`

- Example: Global PTQ (DBF)
- Original PPL: 7.7701
- Quantized + Global PTQ PPL: 37.1946
- Result: the model was saved to ./tinyllama-dbf-globalptq.

## No. 15: `vllm_inference/example_gptq_vllm_inference.py`

- Example: vLLM inference (GPTQ)
- Result: vLLM inference completed successfully.
- Prompt: Explain what post-training quantization is in one sentence:
- Response: the model produced a plausible but imperfect explanation of post-training quantization.
- Prompt: The capital of France is
- Response: the model answered `Paris`, but then continued with other capital cities.

## No. 16: `vllm_inference/example_jointq_vllm_inference.py`

- Example: vLLM inference (JointQ)
- Result: vLLM inference completed successfully.
- Prompt: Explain what post-training quantization is in one sentence:
- Response: the model produced a plausible but imperfect explanation.
- Prompt: The capital of France is
- Response: the model answered `Paris.`, then continued with a numbered list of other countries' capitals.

## No. 17: `vllm_inference/example_autobit_vllm_inference.py`

- Example: vLLM inference (AutoBit)
- Result: vLLM inference completed successfully.
- Prompt: Explain what post-training quantization is in one sentence:
- Response: the model produced a non-empty, bullet-style explanation of quantization and post-training quantization, ending mid-sentence.
- Prompt: The capital of France is
- Response: the model answered `Paris.`, then repeatedly generated `The capital of France is Paris.` before ending mid-sentence.
