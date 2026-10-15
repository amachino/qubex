# `QuantumSimulator` クラス

`QuantumSimulator` は、パルスレベルのハミルトニアン解析をオフラインで行うための入口です。
量子系をモデル化し、pulse を与え、実機に接続せずに実験を反復したいときに使います。

## `QuantumSimulator` を使うべき人

- 実機を使わずにパルスレベルのダイナミクスを調べたい研究者
- 実システムへ移る前にモデルの振る舞いを確かめたいユーザー
- 較正やパルス設計をオフラインで試したいチーム

## `QuantumSimulator` でできること

- 量子ビット、共振器、結合系に対するパルスレベルのハミルトニアンシミュレーション
- Qubex の `Pulse` オブジェクトをオフライン解析にそのまま再利用
- 実機時間を使う前に較正フローを試す安全な経路

## target unitary を作る

`QuantumSystem.unitary()` を使うと、system の物理 Hilbert 空間に target を構成
できます。key は順序付きの object label です。ハイフン区切りの key は tuple の
簡略記法なので、`"Q04-Q01"` と `("Q04", "Q01")` は同じ向きになります。

```python
from qxsimulator import gates

target = system.unitary(
    {
        "Q04": "X",
        "Q01": "H",
    }
)
cz_target = system.unitary({"Q04-Q01": "CZ"})
ef_target = system.unitary(
    {"Q04": gates.X},
    levels={"Q04": (1, 2)},
)
```

文字列は名前付きの static gate として解決されます。`qubex.clifford` で使う gate
名に加えて、一般的な 1-qubit / 2-qubit gate、`ZX90`、`BSWAP`、
`SQRT_BSWAP` を利用できます。引数を持つ gate は、Hermitian 生成子から構成します。

```python
x_rotation = gates.rotation(gates.X, angle)
zx_rotation = gates.rotation(gates.ZX, angle)
exchange_rotation = gates.rotation((gates.XX + gates.YY) / 2, angle)
bswap_rotation = gates.rotation((gates.YY - gates.XX) / 2, angle)
```

`rotation(generator, angle)` は `exp(-1j * angle * generator / 2)` を計算する
ため、相互作用の規格化と符号は生成子の式で決まります。同じ mapping 内の
operation は互いに異なる object を target にする必要があります。逐次 operation
は、個別に作った unitary を積算してください。

小さい gate は、既定では先頭の同じ数の物理 level に embed され、その subspace
外では identity として働きます。別の物理 level に配置する場合は `levels` を明示
します。

## fidelity の評価空間を選ぶ

`process_fidelity()` と `average_gate_fidelity()` は、比較前に物理 propagator を
選択した tensor-product subspace へ射影します。

- `levels="computational"` が既定値で、各 object の level 0 と 1 を選びます。
- `levels="full"` は、すべての物理 level を評価します。
- mapping は指定 object の選択だけを上書きし、未指定 object は computational
  subspace のままです。例えば `levels={"Q04": (1, 2)}` とします。

target には、`system.unitary()` が受け取る object label の mapping をそのまま
渡せます。

```python
fidelity = simulator.average_gate_fidelity(
    controls,
    target_unitary={"Q04-Q01": "CZ"},
)
ef_fidelity = simulator.average_gate_fidelity(
    controls,
    target_unitary={"Q04": gates.X},
    levels={"Q04": (1, 2)},
)
```

mapping target に object ごとの `levels` を指定すると、同じ level 選択を物理空間
への embed と fidelity 評価の両方に使用します。`Qobj` target は、選択後の
subspace と同じ次元、または物理 system 全体の次元で指定できます。full-system
target は選択した評価空間を保存する必要があります。average gate fidelity では、
選択空間の外へ出た確率を leakage として失敗に数えます。

## 推奨する進み方

1. [インストール](../getting-started/installation.md) で Qubex を入れる
2. 必要に応じて共有のパルスシーケンスモデルを確認する: [パルスシーケンスの組み方](../pulse-sequences/index.md)
3. [QuantumSimulator サンプルワークフロー](examples.md) から notebook を始める

`QuantumSimulator` notebook を始めるのに、実機向けの設定ファイルは不要です。

## 次のような場合は `Experiment` を選ぶ

- 実機上で実験を実行したい
- 測定結果や実機ベースの読み出しが必要
- 接続、実行、解析まで含む高レベルワークフローを使いたい

その場合は [`Experiment` クラス](../experiment/index.md) を参照してください。
