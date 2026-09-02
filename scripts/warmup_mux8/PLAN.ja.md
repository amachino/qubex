# mux 8 昇温キャラクタリゼーション実験計画

- 対象: mux 8 の全量子ビット(例: 64Q チップなら Q32–Q35。実際のラベルは `Experiment(muxes=[8]).qubit_labels` で確認)
- 実行物: `scripts/warmup_mux8/run_warmup_mux8.py`(実体は `qubex.contrib.warmup_campaign`)
- 想定期間: 昇温開始 〜 全信号消失(量子ビットは ~1 K 目安、Nb 系共振器は ~9 K 目安まで)

## 1. 目的

昇温を「温度の掃引」として利用し、次を単一の自動キャンペーンで取得する。

1. **量子ビット実効温度 T_eff(t)** — e-f Rabi 振幅比による熱励起確率 p_ex から換算(読み出し誤差に一次で不感)。低温側の飽和値と冷凍機温度計への合流点から残留加熱を定量化する。
2. **Γ1(t) = 1/T1** — 熱的準粒子による指数的増大から超伝導ギャップ Δ を量子ビットごとに抽出し、接合抵抗由来の Ambegaokar–Baratoff 推定(`params/superconducting_gap.yaml`)と照合する。
3. **T2 echo / T2\*(t)** — 読み出し共振器の熱光子デファージングと TLS 起因の変化。
4. **f01(t)** — 運動インダクタンス変化・TLS 分散シフトによる周波数シフト(Ramsey の bare_freq で追跡)。
5. **f_r(t), κ_i(t), κ_e(t)** — 共振器反射測定。低温側 TLS 項と高温側 Mattis–Bardeen 項の競合。量子ビット消失後も継続する主データ。
6. **単発 IQ 分布のスナップショット** — 熱励起増加によるヒストグラムの重み変化と融合の記録(npz 保存、事後解析用)。

## 2. 前提・事前準備(昇温開始前に完了)

- ge π パルス・ef π パルス・Rabi パラメータの較正が最新であること(ef は thermal ステップにのみ必要)。
- **dry-run**: 冷えた状態で全チェーンを検証する。

  ```bash
  uv run python scripts/warmup_mux8/run_warmup_mux8.py --system-id <SYSTEM_ID> --dry-run
  ```

  1 サイクル(~10 分)が回り、`warmup_log.jsonl` の各ステップが `"status": "ok"` であることを確認する。
- **温度ログ**: qubex は温度計を読まないため、冷凍機側で MXC/Still/4K プレート温度を時刻付きで記録開始する。突合は UTC タイムスタンプで行う(ログの `time` は UTC)。
- ディスク容量: 単発データが数 MB/サイクル程度。12 時間 ×  ~5 サイクル/時で数百 MB を見込む。

## 3. 1 サイクルの構成

| ステップ | qubex API | 取得量 | 目安時間 |
| --- | --- | --- | --- |
| reference_points | `obtain_reference_points` | \|g⟩ 参照位相の更新 | ~10 s |
| rabi_refresh(5 サイクル毎) | `obtain_rabi_params` | Rabi 振幅・位相の再取得 | ~1 min |
| ramsey | `ramsey_experiment` | T2\*, bare_freq(周波数トラッキング) | ~1–2 min |
| t1 | `t1_experiment` | T1 | ~1–2 min |
| t2_echo | `t2_experiment` | T2 echo | ~1–2 min |
| thermal | `measure_thermal_excitation` | p_ex → T_eff | ~2–4 min |
| single_shot | `measure_state_distribution` | 単発 IQ(npz) | ~30 s |
| reflection | `measure_reflection_coefficient` | f_r, κ_e, κ_i | ~1–2 min/共振器 |

合計 ~10–15 分/サイクル。`--cycle-interval 0`(既定)で連続実行し、温度分解能はサイクル時間で決まる。

## 4. フェーズ構成と自動フォールバック

- **Phase I(量子ビット存命)**: 全ステップを実行。コヒーレンス 3 ステップ(ramsey/t1/t2_echo)が 3 サイクル連続で全滅した量子ビットは `qubit_lost` として量子ビット系ステップから除外される(thermal の失敗はカウント対象外。ef 較正欠落で誤って除外しないため)。
- **Phase II(共振器のみ)**: 以降は reflection のみ継続。反射フィットも 3 サイクル連続で失敗した共振器は `resonator_lost`。
- **終了条件**: (a) `<output_dir>/STOP` ファイルの作成、(b) `--max-hours`(既定 12 h)経過、(c) 全信号消失、(d) Ctrl-C(グレースフル終了、ログは保全)。

## 5. 実行手順(当日)

1. 昇温開始時刻が決まり次第、可能ならその 30 分前に起動しベースラインを 2–3 サイクル確保する(開始が昇温に数分遅れても支障はない)。
2. 起動(tmux / nohup 推奨):

   ```bash
   nohup uv run python scripts/warmup_mux8/run_warmup_mux8.py --system-id <SYSTEM_ID> \
       > warmup_mux8.out 2>&1 &
   ```

3. 監視: `<output_dir>/summary.json` に最新値(サイクル番号、生存フラグ、追跡周波数)が常時反映される。`tail -f <output_dir>/warmup_log.jsonl` で逐次確認。
4. 停止: `touch <output_dir>/STOP`(現サイクル完了後に停止)。
5. 終了後、`<output_dir>/figures/` に各メトリクスの HTML 図が自動生成される。

## 6. パラメータ既定値と調整指針

- **周波数トラッキング**: 各サイクルの Ramsey bare_freq を次サイクルの駆動周波数に採用(較正値から ±50 MHz 以内のみ受理、`max_frequency_shift`)。サイクル間のドリフトがこれを超えると追跡が外れるため、Phase I では昇温レートを緩やかに保つのが望ましい。
- **thermal_shots**(既定 65536): p_ex 分解能はおよそ 1/√N。ベース温度の p_ex ~0.1–1 % を分解するには 65536 以上を推奨。昇温後半は 4096 程度でも十分なので、時間を優先する場合は `--thermal-shots` を下げる。
- **reflection**: `--electrical-delay <ns>` を dry-run で得た値に固定すると、サイクル毎の遅延測定を省略でき短縮になる。
- ステップの除外は `--skip-steps thermal single_shot` のように指定する。

## 7. 解析計画

- 図生成: `qubex.contrib.plot_warmup_log(output_dir)`(t1, t2_echo, t2_star, f_ge, p_ex, t_eff, f_r, kappa_in, kappa_ex)。
- **温度突合**: 冷凍機ログと UTC 時刻で内挿結合し、横軸を時刻から MXC 温度に変換する。
- **T_eff vs T_MXC**: 低温側プラトー(残留加熱)と合流点を読む。
- **ギャップフィット**: Γ1(T) − Γ1(T→0) ∝ √T · exp(−Δ/k_B T) で各量子ビットの Δ を抽出し、AB 推定値と比較。f01(T) シフトでも独立に Δ を確認。
- **T1 非単調性**: 100–200 mK 帯での一時的な T1 改善(TLS 飽和)の有無。
- **f_r(T)**: TLS 項 + Mattis–Bardeen 項の同時フィット。κ_i(T) から Q_i(T)。
- **単発 npz**: サイクル毎に g/e ブロブの分離度・重みを再構成し、p_ex の独立検証と読み出し忠実度の温度限界を評価。

## 8. リスクと対処

| リスク | 挙動 | 対処 |
| --- | --- | --- |
| ef パルス未較正 | thermal のみ failed 記録、他は継続 | `--skip-steps thermal` で明示除外可 |
| 急峻な周波数ドリフト | Ramsey フィット失敗 → 追跡停止 | rabi_refresh で緩和。連続失敗時は自動で qubit_lost |
| JPA/ポンプの温度不安定 | 全ステップの SNR 低下 | reflection の κ_e・振幅変化で切り分け |
| プロセスクラッシュ | JSONL は追記型で途中まで保全 | 再実行(新 output_dir)。ログは時刻で連結可能 |

## 9. チェックリスト

- [ ] dry-run 全ステップ ok(前日まで)
- [ ] 冷凍機温度ログの記録開始
- [ ] tmux 上でキャンペーン起動、cycle 1 の完了を確認
- [ ] `summary.json` の `tracked_frequencies` 更新を確認
- [ ] 昇温開始時刻を UTC で記録(ログとの突合キー)
