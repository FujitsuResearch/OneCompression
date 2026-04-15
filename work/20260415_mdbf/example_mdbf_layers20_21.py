"""
MDBF 移植確認用スクリプト（層 20・21 限定）

QEP-DEV の experiments/lowbit_formats/conf/base.yaml と同条件で layers.20・layers.21 のみを
MDBF 量子化し、Dequantized PPL を計測する。

QEP-DEV との対応:
  - dataset=wikitext2 / nsamples=32 / seqlen=2048 / seed=0   → CalibrationConfig
  - gqep=true                                                 → Runner(qep=True)
  - percdampgqep=0.01 / perccorr=0.5                         → QEPConfig
  - apply_mlp_correct=false                                   → QEPConfig(exclude_layer_keywords=["mlp.*"])
  - strategy.cut_first_layers=20, cut_topk=0                 → MDBF(include_layer_keywords=["layers.20","layers.21"])
  - bits=1.0 / l=1 / P=2 / svd_mode=svd                     → MDBF
  - use_admm / admm_iters=100 / admm_inner_iters=3 / reg=0.03 → MDBF

揃えられない条件（既知の差異）:
  - サンプリング方式: QEP-DEV は全文結合→ランダムスライス(seed=0)、
                      OneComp は全文結合→先頭 32 チャンク固定 (concat_chunk_align)
  - QEP 適用方式:     QEP-DEV は量子化前に重みを補正、OneComp は前層誤差を次層に伝播

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
# 統計ファイルの保存先（スクリプトと同階層）
STATS_DIR = os.path.dirname(os.path.abspath(__file__))

# QEP-DEV: apply_mlp_correct=false 相当（MLP 全体を QEP からスキップ）
_QEP_CONFIG = QEPConfig(
    percdamp=0.01,
    perccorr=0.5,
    exclude_layer_keywords=["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
)

# キャリブレーション設定（QEP-DEV の dataset=wikitext2, nsamples=32, seqlen=2048, seed=0 に対応）
_CALIB_CONFIG = CalibrationConfig(
    calibration_dataset="wikitext2",
    max_length=2048,
    num_calibration_samples=32,
    strategy="concat_chunk_align",
    seed=0,
)


def run_case(use_admm: bool, label: str):
    """
    1 ケース（no-ADMM または ADMM-100）を実行して Dequantized PPL を返す。

    Args:
        use_admm: ADMM 最適化を使用するか
        label: ケース名（表示用・統計ファイル名に使用）

    Returns:
        dequantized_ppl (float)
    """
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

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
        include_layer_keywords=["layers.20", "layers.21"],
    )

    runner = Runner(
        model_config=model_config,
        quantizer=mdbf,
        qep=True,
        qep_config=_QEP_CONFIG,
        calibration_config=_CALIB_CONFIG,
    )

    runner.run()

    # 量子化誤差を表示
    # runner.print_quantization_results()

    # 量子化統計情報を保存
    stats_path = os.path.join(STATS_DIR, f"mdbf_layers20_21_{label}_statistics.json")
    runner.save_quantization_statistics(stats_path)
    print(f"Quantization statistics saved to: {stats_path}")

    # PPL を計測（dequantized のみ）
    original_ppl, dequantized_ppl, _ = runner.calculate_perplexity(
        original_model=True,
        dequantized_model=True,
        quantized_model=False,
    )

    print(f"Original model perplexity    : {original_ppl:.2f}")
    print(f"Dequantized model perplexity : {dequantized_ppl:.2f}")

    del runner
    return dequantized_ppl


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    setup_logger()

    # SVD 初期化の randn_like が結果に影響するため、再現性のためシードを固定
    torch.manual_seed(0)

    cases = [
        (False, "no-ADMM"),
        (True,  "ADMM-100"),
    ]

    summary = []
    for use_admm, label in cases:
        ppl = run_case(use_admm=use_admm, label=label)
        summary.append((label, ppl))

    print("\n" + "="*60)
    print("  Summary  (layers 20 & 21 only)")
    print("="*60)
    print(f"  {'':12s}  {'Dequant PPL':>12s}")
    print(f"  {'-'*12}  {'-'*12}")
    for label, ppl in summary:
        print(f"  {label:12s}  {ppl:>12.2f}")
    print("="*60)
