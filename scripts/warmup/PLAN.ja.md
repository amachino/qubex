# mux 7・9 昇温キャラクタリゼーション実験計画

- 対象: mux 7 と mux 9 の全量子ビット(計 8 量子ビット。実際のラベルは `Experiment(muxes=[7, 9]).qubit_labels` で確認)
- 実行物: `scripts/warmup/run_warmup.py`(既定 `--muxes 7 9`。実体は `qubex.contrib.warmup_campaign`)
- 想定期間: 昇温開始 〜 全信号消失(量子ビットは ~1 K 目安、Nb 系共振器は ~9 K 目安まで)
- **制約: mux 8 は bswap 実験が並行稼働中のため、絶対に触れない**(下記 0 章)

## 0. mux 8 保護(最優先)

qubex の次の操作は**ボックス単位**で作用するため、mux 7/9 の量子ビットしか指定していなくても、制御ボックスを mux 8 と共有していれば bswap 実験を破壊する。

- `reset_awg_and_capunits`(反射測定などが内部で呼ぶ): 対象量子ビットが属する**ボックス全体**の AWG/キャプチャユニットを初期化する
- `connect()` の既定動作: 選択ボックスのクロックをキック(再同期)する

このためランナーは起動時に次を機械的に検証し、**一つでも該当すればハードウェアに触れる前に終了する**(上書きオプションは用意していない)。

1. `--muxes` に禁止 mux(既定 `--forbidden-muxes 8`)が含まれていない
2. 選択した量子ビットが禁止 mux に属していない
3. 選択した量子ビットのボックス(`exp.box_ids`)が、禁止 mux の量子ビットのボックスと**一つも重ならない**(`qubex.contrib.check_mux_isolation`)
4. `--targets` 指定がある場合、すべて選択 mux 内の量子ビットである

加えて `connect()` はクロック再同期を**行わない**(`--sync-clocks` を明示した場合のみ実施。実施しても対象は選択ボックスのみで、上記 3 により mux 8 のボックスは含まれない)。接続後に `check_status()` でクロック状態をログに出すので、非同期の警告が出た場合のみ `--sync-clocks` を検討する。

`--preflight` はこの隔離チェックと較正チェックをオフラインで実行して終了するため、**まず `--preflight` を実行し、`isolated : True` を確認してから**先に進むこと。もし `shared boxes` が空でなければ、mux 7/9 のいずれかが mux 8 とボックスを共有しており、bswap 実験と同時に昇温実験を行うことはできない(共有していない mux に変更するか、bswap 実験の終了を待つ)。

## 1. 目的

昇温を「温度の掃引」として利用し、次を単一の自動キャンペーンで取得する。

1. **量子ビット実効温度 T_eff(t)** — e-f Rabi 振幅比による熱励起確率 p_ex から換算(読み出し誤差に一次で不感)。低温側の飽和値と冷凍機温度計への合流点から残留加熱を定量化する。2 mux 分の配線系統で残留加熱の系統差も比較できる。
2. **Γ1(t) = 1/T1** — 熱的準粒子による指数的増大から超伝導ギャップ Δ を量子ビットごとに抽出し、接合抵抗由来の Ambegaokar–Baratoff 推定(`params/superconducting_gap.yaml`)と照合する。8 量子ビット分のギャップマップが得られる。
3. **T2 echo / T2\*(t)** — 読み出し共振器の熱光子デファージングと TLS 起因の変化。
4. **f01(t)** — 運動インダクタンス変化・TLS 分散シフトによる周波数シフト(Ramsey の bare_freq で追跡)。
5. **f_r(t), κ_i(t), κ_e(t)** — 共振器反射測定。低温側 TLS 項と高温側 Mattis–Bardeen 項の競合。量子ビット消失後も継続する主データ。
6. **単発 IQ 分布のスナップショット** — 熱励起増加によるヒストグラムの重み変化と融合の記録(npz 保存、事後解析用)。

## 2. 事前準備(昇温開始前に完了)

1. **較正の確認(オフライン、ハードウェア不要)**:

   ```bash
   uv run python scripts/warmup/run_warmup.py --system-id <SYSTEM_ID> --preflight
   ```

   まず mux 隔離チェック(0 章)の結果が表示され、`isolated : True` でなければここで終了する。続いて量子ビットごとに ge 周波数・読み出し周波数・Rabi パラメータ・ge π パルス・ef π パルスの有無と、各ステップの実行可否が表示される。`NG` があれば該当較正を先に実施する(ef π パルスは thermal ステップにのみ必要)。

2. **dry-run(ハードウェア検証)**: 冷えた状態で全チェーンを 1 サイクル回す。

   ```bash
   uv run python scripts/warmup/run_warmup.py --system-id <SYSTEM_ID> --dry-run
   ```

   `warmup_log.jsonl` の各ステップが `"status": "ok"` であること、`figures/` に図が出ることを確認する。ここで得た反射測定の electrical delay を控えておくと本番で `--electrical-delay` に使える。

3. **温度ログ**: qubex は温度計を読まないため、冷凍機側で MXC/Still/4K プレート温度を時刻付きで記録開始する。突合は UTC タイムスタンプで行う(ログの `time` は UTC)。
4. ディスク容量: 単発データが数 MB/サイクル程度。12 時間で数百 MB を見込む。

## 3. 1 サイクルの構成

| ステップ | qubex API | 取得量 | 目安時間(8 量子ビット) |
| --- | --- | --- | --- |
| reference_points | `obtain_reference_points` | \|g⟩ 参照位相の更新 | ~10 s |
| rabi_refresh(5 サイクル毎) | `obtain_rabi_params` | Rabi 振幅・位相の再取得 | ~1–2 min |
| ramsey | `ramsey_experiment` | T2\*, bare_freq(周波数トラッキング) | ~2–3 min |
| t1 | `t1_experiment` | T1 | ~2–3 min |
| t2_echo | `t2_experiment` | T2 echo | ~2–3 min |
| thermal | `measure_thermal_excitation` | p_ex → T_eff | ~4–6 min(逐次) |
| single_shot | `measure_state_distribution` | 単発 IQ(npz) | ~30 s |
| reflection | `measure_reflection_coefficient` | f_r, κ_e, κ_i | ~1–2 min × 8(逐次) |

掃引系(ramsey/t1/t2_echo)は qubex が量子ビットを 2 サブグループ(index%4 ∈ {0,3} と {1,2})に分けて同時掃引するため、mux 数を増やしても所要時間はほぼ変わらない。逐次実行の thermal と reflection が支配的で、合計 ~20–30 分/サイクル。時間分解能を上げたい場合は次で短縮できる。

- `--electrical-delay <ns>`: サイクル毎の遅延測定を省略(reflection を大幅短縮)
- `--reflection-df 0.001`: 反射掃引の点数を半減
- `--thermal-shots 16384`: 昇温後半(p_ex が数 % 以上)では十分な分解能

## 4. フェーズ構成と自動フォールバック

- **Phase I(量子ビット存命)**: 全ステップを実行。コヒーレンス 3 ステップ(ramsey/t1/t2_echo)が 3 サイクル連続で全滅した量子ビットは `qubit_lost` として量子ビット系ステップから除外される(thermal の失敗はカウント対象外。ef 較正欠落で誤って除外しないため)。
- **Phase II(共振器のみ)**: 以降は reflection のみ継続。反射フィットも 3 サイクル連続で失敗した共振器は `resonator_lost`。
- **終了条件**: (a) `<output_dir>/STOP` ファイルの作成、(b) `--max-hours`(既定 12 h)経過、(c) 全信号消失、(d) Ctrl-C(グレースフル終了、ログは保全)。

## 5. 実行手順(当日)

1. 昇温開始時刻が決まり次第、可能ならその 30 分前に起動しベースラインを 1–2 サイクル確保する(開始が昇温に数分遅れても支障はない)。
2. 起動(tmux / nohup 推奨):

   ```bash
   nohup uv run python scripts/warmup/run_warmup.py --system-id <SYSTEM_ID> \
       --electrical-delay <dry-run で得た値> > warmup.out 2>&1 &
   ```

   起動時に隔離チェックと preflight の結果が表示される。隔離チェックに失敗すれば接続前に終了する。較正の `NG` は該当ステップを failed として記録しつつ継続する。クロック再同期は行わない(`check_status()` の出力で非同期が報告された場合のみ `--sync-clocks` を追加して再起動する)。
3. 監視: `<output_dir>/summary.json` に最新値(サイクル番号、生存フラグ、追跡周波数)が常時反映される。`tail -f <output_dir>/warmup_log.jsonl` で逐次確認。
4. 停止: `touch <output_dir>/STOP`(現サイクル完了後に停止)。
5. 終了後、`<output_dir>/figures/` に各メトリクスの HTML 図が自動生成される。

## 6. パラメータ既定値と調整指針

- **周波数トラッキング**: 各サイクルの Ramsey bare_freq を次サイクルの駆動周波数に採用(較正値から ±50 MHz 以内のみ受理、`max_frequency_shift`)。サイクル間のドリフトがこれを超えると追跡が外れるため、Phase I では昇温レートを緩やかに保つのが望ましい。
- **thermal_shots**(既定 65536): p_ex 分解能はおよそ 1/√N。ベース温度の p_ex ~0.1–1 % を分解するには 65536 以上を推奨。
- ステップの除外は `--skip-steps thermal single_shot` のように指定する。特定の量子ビットに絞る場合は `--targets Q28 Q29` のように指定する。

## 7. 解析計画

- 図生成: `qubex.contrib.plot_warmup_log(output_dir)`(t1, t2_echo, t2_star, f_ge, p_ex, t_eff, f_r, kappa_in, kappa_ex)。
- **温度突合**: 冷凍機ログと UTC 時刻で内挿結合し、横軸を時刻から MXC 温度に変換する。
- **T_eff vs T_MXC**: 低温側プラトー(残留加熱)と合流点を読む。mux 7 と 9 で系統差があれば配線・アッテネータ構成の違いを疑う。
- **ギャップフィット**: Γ1(T) − Γ1(T→0) ∝ √T · exp(−Δ/k_B T) で各量子ビットの Δ を抽出し、AB 推定値と比較。f01(T) シフトでも独立に Δ を確認。
- **T1 非単調性**: 100–200 mK 帯での一時的な T1 改善(TLS 飽和)の有無。
- **f_r(T)**: TLS 項 + Mattis–Bardeen 項の同時フィット。κ_i(T) から Q_i(T)。
- **単発 npz**: サイクル毎に g/e ブロブの分離度・重みを再構成し、p_ex の独立検証と読み出し忠実度の温度限界を評価。

## 8. リスクと対処

| リスク | 挙動 | 対処 |
| --- | --- | --- |
| mux 8 とのボックス共有 | ランナーが接続前に終了 | 共有のない mux に変更するか、bswap 実験終了後に実施 |
| ef パルス未較正 | thermal のみ failed 記録、他は継続 | preflight で事前検出。`--skip-steps thermal` で明示除外可 |
| 急峻な周波数ドリフト | Ramsey フィット失敗 → 追跡停止 | rabi_refresh で緩和。連続失敗時は自動で qubit_lost |
| JPA/ポンプの温度不安定 | 全ステップの SNR 低下 | reflection の κ_e・振幅変化で切り分け |
| サイクルが長すぎる | 温度分解能の低下 | `--electrical-delay` 固定、`--reflection-df` 拡大、`--thermal-shots` 削減 |
| プロセスクラッシュ | JSONL は追記型で途中まで保全 | 再実行(新 output_dir)。ログは時刻で連結可能 |

## 9. チェックリスト

- [ ] `--preflight` で `isolated : True`(mux 8 とボックス共有なし)を確認(前日まで)
- [ ] `--preflight` で全量子ビット・全ステップ ok(前日まで)
- [ ] `--dry-run` 全ステップ ok、electrical delay を控える(前日まで)
- [ ] 冷凍機温度ログの記録開始
- [ ] tmux 上でキャンペーン起動、cycle 1 の完了を確認
- [ ] `summary.json` の `tracked_frequencies` 更新を確認
- [ ] 昇温開始時刻を UTC で記録(ログとの突合キー)
