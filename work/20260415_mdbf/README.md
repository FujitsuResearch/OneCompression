# MDBF 移植確認テスト

## ファイル構成

```
work/20260415_mdbf/
  example_mdbf.py              ← MDBF 動作確認スクリプト（全機能テスト）
  example_mdbf_layers20_21.py  ← QEP-DEV との PPL 比較用スクリプト（層 20・21 限定）
  README.md                    ← 本ファイル
```

## 実行方法

**必ず `CM_OneCompression/` ディレクトリから実行すること**（パッケージの相対 import のため）。

```bash
cd /home/cm/FJ/mhsuzu/OneComp/CM_OneCompression

# 動作確認スクリプト
python work/20260415_mdbf/example_mdbf.py

# QEP-DEV 比較スクリプト（層 20・21 限定）
python work/20260415_mdbf/example_mdbf_layers20_21.py
```

> **uv 仮想環境を使う場合:**  
> `uv sync` 後に torch が意図せず cu130 ビルドへ更新されることがある。  
> その場合は `--no-sync` を付けて実行する。
> ```bash
> uv run --no-sync work/20260415_mdbf/example_mdbf.py
> uv run --no-sync work/20260415_mdbf/example_mdbf_layers20_21.py
> ```

## テスト関数一覧（example_mdbf.py）

`example_mdbf.py` の `main()` ブロックで実行する関数を切り替える。  
不要な行をコメントアウトして使う。

| 関数 | 内容 | Phase |
|---|---|---|
| `test_basic_quantization()` | 量子化 → PPL 出力（全 22 層、動作確認用） | 1, 2 |
| `test_inspect_results()` | MDBFResult の shape / actual_bpw / r を全レイヤー出力 | 1 |
| `test_save_model()` | `save_quantized_model()` を呼ぶ（Phase 3 未実装時はエラーを catch して表示） | 2 |
| `test_inference_layer()` | `create_inference_layer()` で `MultipathMSVIDLinear` を生成し forward を確認 | 2 |
| `test_compare_with_qep_dev()` | **QEP-DEV 実験と同条件**（7/22 層）で PPL を比較（→ 結果は上記参照） | 1 |

### デフォルト実行（`main()` 末尾）

```python
# test_basic_quantization()     # 量子化 + PPL（最小確認、全レイヤー量子化）
# test_inspect_results()        # MDBFResult の shape / bpw / r を詳細出力
# test_save_model()             # save_quantized_model() の動作確認
# test_inference_layer()        # MultipathMSVIDLinear の forward 確認
test_compare_with_qep_dev()   # QEP-DEV 実験と同条件（7/22層）の PPL 比較
```

## MDBF パラメータ設定箇所（example_mdbf.py）

`example_mdbf.py` の `_make_runner()` 内の `MDBF(...)` を変更する。

```python
mdbf = MDBF(
    target_bits=1.0,     # 目標 BPW
    l=1,                 # Multi-scale ランク
    P=2,                 # パス数（1 or 2）
    svd_mode="svd",      # 初期化モード（"svd" or "svd_llm"）
    use_admm=True,       # ADMM 最適化
    admm_iters=260,
    admm_inner_iters=3,
    admm_reg=0.03,
    use_gradient_refine=False,
    activation_aware=False,
)
```

---

## QEP-DEV 比較スクリプト（example_mdbf_layers20_21.py）

QEP-DEV の `experiments/lowbit_formats/conf/base.yaml` と同条件で
**layers.20・layers.21 のみ**を MDBF 量子化し、Dequantized PPL を計測する。

### QEP-DEV との条件対応

| 内容 | QEP-DEV | OneComp |
|---|---|---|
| キャリブレーションデータ | `dataset: wikitext2` | `CalibrationConfig(calibration_dataset="wikitext2")` |
| シーケンス長 | `seqlen: 2048` | `CalibrationConfig(max_length=2048)` |
| サンプル数 | `nsamples: 32` | `CalibrationConfig(num_calibration_samples=32)` |
| GQEP | `gqep: true` | `Runner(qep=True)` |
| MLP QEP スキップ | `apply_mlp_correct: false` | `QEPConfig(exclude_layer_keywords=["mlp.*"])` |
| 量子化層 | `cut_first_layers=20, cut_topk=0` | `MDBF(include_layer_keywords=["layers.20","layers.21"])` |

**揃えられない差異（既知）:**
- サンプリング方式: QEP-DEV はランダムスライス、OneComp は先頭固定 (concat_chunk_align)
- QEP 適用方式: QEP-DEV は量子化前に重み補正、OneComp は前層誤差を次層に伝播

### 出力

- **統計 JSON**: スクリプトと同階層に保存
  - `mdbf_layers20_21_no-ADMM_statistics.json`
  - `mdbf_layers20_21_ADMM-100_statistics.json`
- **PPL**: 実行末尾の Summary テーブルに Original PPL と Dequantized PPL を表示

### PPL 検証結果（TinyLlama-1.1B、層 20・21 のみ量子化）

| ケース | QEP-DEV PPL | OneComp PPL | 差 |
|---|---:|---:|---|
| Original (FP16) | — | 7.77 | — |
| no-ADMM | 29.43 | 37.26 | +7.83 |
| ADMM-100 | 12.21 | 14.07 | +1.86 |

---

## PPL 検証結果（example_mdbf.py / TinyLlama-1.1B、7/22 層量子化）

### 条件

- QEP-DEV と同じ 7/22 層のみ量子化（layer 3,4,5,9,10,12,20）
- `qep=True`, `QEPConfig(exclude_layer_keywords=["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"])`
  - QEP-DEV の `gqep=True, apply_mlp_correct=false` に相当

### 結果

| 設定 | OneComp PPL | QEP-DEV PPL |
|---|---|---|
| no-ADMM + QEP | ~323 | 319.06 ✓ |
| ADMM-100 + QEP | ~31.6 | 23.93 |

no-ADMM は QEP-DEV とほぼ一致。ADMM ありの差は QEP の適用順序の違いによる
（QEP-DEV は量子化**前**に重み補正、OneComp は量子化**後**に誤差伝播）。

### アルゴリズム実装の正確性

`initialize_msvid` / `optimize_msvid_admm` の出力が QEP-DEV と bit-for-bit 一致することを確認済み
（同一入力での再構成誤差が完全一致）。

### `test_basic_quantization()` の PPL が高い理由

全 22 層を 1 BPW で量子化するため。1 BPW での SVD 初期化後の重み再構成誤差は約 70% あり、
22 層すべてに誤差が乗ると PPL が数万〜数十万になるのは正常な動作。

## 注意事項

- **Phase 3（モデルロード対応）は未実装**のため、`test_save_model()` は `save_quantized_model()` が safetensors 保存でエラーを出す可能性がある。エラーは catch して表示するため、スクリプト自体は止まらない。
- テストモデルは `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`（HuggingFace から自動ダウンロード）。モデルID は `example_mdbf.py` 冒頭の `MODEL_ID` で変更可。
