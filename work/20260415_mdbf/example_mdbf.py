"""
MDBF 移植確認用テストスクリプト

各テスト関数を単独で実行できるよう関数化してある。
不要な関数は main() 末尾のコメントアウトで除外して使う。

関数一覧:
  test_basic_quantization()   : 量子化 → PPL 計測（全 22 層量子化、最小確認）
  test_inspect_results()      : MDBFResult の中身（shape / bpw / r）を出力
  test_save_model()           : 量子化 → save_quantized_model()（Phase 3 未実装時はエラー catch）
  test_inference_layer()      : create_inference_layer() (MultipathMSVIDLinear) の forward 確認
  test_compare_with_qep_dev() : QEP-DEV 実験と同条件（7/22 層）で PPL を比較

"""

import os
import torch
from onecomp import MDBF, CalibrationConfig, ModelConfig, Runner, setup_logger
from onecomp.qep import QEPConfig

# ---------------------------------------------------------------------------
# 共通設定
# ---------------------------------------------------------------------------
MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
DEVICE   = "cuda:0"
# スクリプトと同階層に保存する
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tinyllama_mdbf_1bit")

# QEP-DEV の apply_mlp_correct=false 相当（MLP 全体を QEP からスキップ）
_QEP_CONFIG = QEPConfig(
    exclude_layer_keywords=["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
)


def _make_runner(use_admm: bool = False, use_qep: bool = False) -> Runner:
    """量子化済み Runner を生成して返す（各テスト関数の共通処理）。"""
    model_config = ModelConfig(model_id=MODEL_ID, device=DEVICE)
    mdbf = MDBF(
        target_bits=1.0,
        l=1,
        P=2,
        svd_mode="svd",
        use_admm=use_admm,
        admm_iters=100,
        admm_inner_iters=3,
        admm_reg=0.03,
        use_gradient_refine=False,
        activation_aware=False,
    )
    calib_config = CalibrationConfig(
        calibration_dataset="wikitext2",
        max_length=2048,
        num_calibration_samples=32,
        strategy="concat_chunk_align",
    )
    runner = Runner(
        model_config=model_config,
        quantizer=mdbf,
        qep=use_qep,
        qep_config=_QEP_CONFIG if use_qep else None,
        calibration_config=calib_config,
    )
    runner.run()
    return runner


# ---------------------------------------------------------------------------
# テスト関数
# ---------------------------------------------------------------------------

def test_basic_quantization():
    """
    最小構成の動作確認（全 22 層量子化）。
    - quantize_layer() が MDBFResult を返すこと
    - compute_dequantized_weight() が動作すること（dequantized_model=True）
    - create_inference_layer() が MultipathMSVIDLinear を返すこと（quantized_model=True）

    注意: 全 22 層を 1 BPW で量子化するため PPL は非常に高くなる（QEP-DEV との比較には不向き）。
    """
    print("\n" + "="*60)
    print("[test_basic_quantization]")
    print("="*60)

    runner = _make_runner()

    original_ppl, dequantized_ppl, quantized_ppl = runner.calculate_perplexity(
        original_model=True, dequantized_model=True, quantized_model=True
    )

    print(f"Original model perplexity    : {original_ppl:.2f}")
    print(f"Dequantized model perplexity : {dequantized_ppl:.2f}")
    print(f"Quantized model perplexity   : {quantized_ppl:.2f}")


def test_inspect_results():
    """
    MDBFResult の中身を詳細に出力して確認する。
    - 各レイヤーの actual_bpw / r / P
    - 各テンソルリストの shape
    - compute_dequantized_weight() の shape と dtype
    """
    print("\n" + "="*60)
    print("[test_inspect_results]")
    print("="*60)

    runner = _make_runner()
    results = runner.quantizer.results

    print(f"Total quantized layers: {len(results)}")
    print()

    for layer_name, result in results.items():
        P_actual = len(result.mdbf_A_sign)
        print(f"  [{layer_name}]")
        print(f"    P={P_actual}, actual_bpw={result.actual_bpw:.3f}, r={result.r}")
        print(f"    A_sign shapes : {[t.shape for t in result.mdbf_A_sign]}")
        print(f"    B_sign shapes : {[t.shape for t in result.mdbf_B_sign]}")
        print(f"    A_amp  shapes : {[t.shape for t in result.mdbf_A_amp]}")
        print(f"    B_amp  shapes : {[t.shape for t in result.mdbf_B_amp]}")
        W_deq = result.compute_dequantized_weight()
        print(f"    compute_dequantized_weight() -> shape={W_deq.shape}, dtype={W_deq.dtype}")
        print()


def test_save_model():
    """
    量子化後のモデルを save_quantized_model() で保存する。
    Phase 3 (quantized_model_loader.py への mdbf 対応) が未実装の場合、
    safetensors 保存でエラーが出る可能性があるため try/except で catch する。
    """
    print("\n" + "="*60)
    print("[test_save_model]")
    print("="*60)

    runner = _make_runner()

    print(f"Saving to: {SAVE_DIR}")
    try:
        runner.save_quantized_model(SAVE_DIR)
        print("Saved successfully.")
    except Exception as e:
        print(f"[WARNING] save_quantized_model() raised an exception")
        print(f"  (expected if Phase 3 not yet implemented)")
        print(f"  {type(e).__name__}: {e}")


def test_compare_with_qep_dev():
    """
    QEP-DEV 実験と同条件の比較テスト。

    QEP-DEV の実験条件（experiments/lowbit_formats/conf/base.yaml）:
      - cut_topk=15 により 7/22 層のみ量子化（layers 3,4,5,9,10,12,20）
      - gqep=True, apply_mlp_correct=false（MLP への QEP をスキップ）
      - P=2, l=1, admm_iters=100
      - dataset=wikitext2, nsamples=32, seqlen=2048, seed=0

    OneComp の対応設定:
      - calibration_dataset=wikitext2 train テキスト
      - max_length=2048, num_calibration_samples=32
      - calibration_strategy="concat_chunk_align"（全テキスト結合→先頭32チャンク）
      - calibration_seed=0（Runner デフォルト）
      ※ QEP-DEV は全テキスト結合後にランダムスライス（seed=0）。
        OneComp の concat_chunk_align は先頭から順に切り出すため微差が生じる可能性がある。

    更新済み結果（2026-04-14、max_length=2048, wikitext2）:
      （実行後にここへ記録してください）
    """
    # SVD 初期化の randn_like が結果に影響するため、再現性のためシードを固定する
    torch.manual_seed(0)

    print("\n" + "="*60)
    print("[test_compare_with_qep_dev]")
    print("="*60)

    QEP_DEV_QUANTIZED_LAYERS = [3, 4, 5, 9, 10, 12, 20]
    MODULES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
               "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    include_names = [
        f"model.layers.{idx}.{mod}"
        for idx in QEP_DEV_QUANTIZED_LAYERS
        for mod in MODULES
    ]
    print(f"Quantizing {len(include_names)} modules in {len(QEP_DEV_QUANTIZED_LAYERS)} layers")

    model_config = ModelConfig(model_id=MODEL_ID, device=DEVICE)

    # (use_admm, label, qep_dev_ppl)
    cases = [
        (False, "no-ADMM",  319.06),
        (True,  "ADMM-100",  23.93),
    ]

    calib_config = CalibrationConfig(
        calibration_dataset="wikitext2",
        max_length=2048,
        num_calibration_samples=32,
        strategy="concat_chunk_align",
    )

    summary = []
    for use_admm, label, qep_dev_ppl in cases:
        print(f"\n--- Running {label} ---")
        mdbf = MDBF(
            target_bits=1.0, l=1, P=2, svd_mode="svd",
            use_admm=use_admm, admm_iters=100,
            include_layer_names=include_names,
        )
        runner = Runner(
            model_config=model_config, quantizer=mdbf,
            qep=True, qep_config=_QEP_CONFIG,
            calibration_config=calib_config,
        )
        runner.run()
        _, dequantized_ppl, _ = runner.calculate_perplexity(
            original_model=False, dequantized_model=True, quantized_model=False
        )
        summary.append((label, dequantized_ppl, qep_dev_ppl))
        del runner

    print("\n" + "="*60)
    print("  Summary")
    print("="*60)
    print(f"  {'':12s}  {'OneComp PPL':>12s}  {'QEP-DEV PPL':>12s}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*12}")
    for label, our_ppl, ref_ppl in summary:
        print(f"  {label:12s}  {our_ppl:>12.2f}  {ref_ppl:>12.2f}")
    print("="*60)


def test_inference_layer():
    """
    create_inference_layer() で生成した MultipathMSVIDLinear の forward を確認する。
    ダミー入力で forward() を実行し、出力 shape が正しいかチェックする。
    """
    print("\n" + "="*60)
    print("[test_inference_layer]")
    print("="*60)

    runner = _make_runner()
    results = runner.quantizer.results

    # runner.quantizer にある module_to_name の逆引きで Linear を取得
    name_to_module = {v: k for k, v in runner.quantizer.module_to_name.items()}

    ok_count = 0
    for layer_name, result in results.items():
        linear_module = name_to_module.get(layer_name)
        if linear_module is None:
            print(f"  [{layer_name}] module not found, skip")
            continue

        inf_layer = runner.quantizer.create_inference_layer(
            result=result,
            linear_module=linear_module,
        )

        # ダミー入力で forward
        n, m = result.mdbf_A_sign[0].shape[0], result.mdbf_B_sign[0].shape[1]
        x = torch.randn(1, m, dtype=torch.float16, device="cpu")
        with torch.no_grad():
            y = inf_layer(x)

        expected = (1, n)
        assert y.shape == torch.Size(expected), \
            f"[{layer_name}] output shape mismatch: expected {expected}, got {y.shape}"
        ok_count += 1

    print(f"MultipathMSVIDLinear forward OK: {ok_count}/{len(results)} layers")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    setup_logger()

    # 確認したいテストのコメントアウトを外して実行する
    # test_basic_quantization()     # 量子化 + PPL（最小確認、全レイヤー量子化）
    # test_inspect_results()        # MDBFResult の shape / bpw / r を詳細出力
    # test_save_model()             # save_quantized_model() の動作確認
    # test_inference_layer()        # MultipathMSVIDLinear の forward 確認
    test_compare_with_qep_dev()   # QEP-DEV 実験と同条件（7/22層）の PPL 比較
