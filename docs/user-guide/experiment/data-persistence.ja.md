# データ保存

Qubex には、結果を保存する経路が 2 つあります。解析済みの実験レベルの結果を
残す場合は `ExperimentResult.save()` を使います。測定実行中に生成される
`MeasurementResult` 形式の raw data と metadata を残す場合は
`SystemManager.save_rawdata()` または `set_rawdata_dir()` を使います。

| 用途 | API | 保存されるオブジェクト | 既定の保存先 | 形式 |
| --- | --- | --- | --- | --- |
| 解析済みの実験結果を保存する | `ExperimentResult.save()` | `ExperimentRecord[ExperimentResult]` | `data/` | jsonpickle JSON |
| 特定の実験処理において raw data と metadata を保存する | `exp.system_manager.save_rawdata(...)` | `MeasurementResult` | `.rawdata/` | NetCDF4 `.nc` |
| `execute()` / `measure()` の raw-data 保存を有効にし続ける | `exp.system_manager.set_rawdata_dir(...)` | `MeasurementResult` | ユーザーが指定したディレクトリ | NetCDF4 `.nc` |

## `ExperimentResult` を保存する

多くの高レベル `Experiment` メソッドは `ExperimentResult` を返します。この
オブジェクトは、実験向けに整理された target data と `plot()`、`fit()` などの
helper を持ちます。この解析済み結果オブジェクトを残したいときは、結果に対して
`save()` を呼びます。

```python
result = exp.obtain_rabi_params(targets=["Q00"], n_shots=1024)

record = result.save(
    name="q00_rabi",
    description="Rabi calibration for Q00 after frequency update.",
)

print(record.file_name)
```

この convenience method は `ExperimentRecord` を作成し、ただちに保存し、その
record を返します。ファイルは `data/` 以下に
`YYYYMMDD_<name>_<counter>.json` という名前で書かれます。同じ日付・同じ名前の
ファイルを上書きしないよう、counter は自動で増えます。

読み戻しには `ExperimentRecord.load()` または `Experiment` の convenience
method を使います。

```python
loaded = exp.load_record(record.file_name)
restored_result = loaded.data
```

既定以外のディレクトリに保存したい場合は、`ExperimentRecord` を直接使います。

```python
from qubex.experiment.models import ExperimentRecord

record = ExperimentRecord(
    data=result,
    name="q00_rabi",
    description="Rabi calibration for Q00 after frequency update.",
)
record.save(data_path="results/experiment-records")
```

### `ExperimentRecord` の注意点

- `ExperimentResult.save()` が保存するのは高レベルな結果オブジェクトです。
  実験の途中で生成されたすべての `MeasurementResult` を自動で保存するわけでは
  ありません。
- ファイル名には `name` がそのまま入ります。空白や path separator を避け、
  短く filesystem-safe な名前を使ってください。
- `ExperimentRecord` ファイルは jsonpickle で encode された Python object
  graph です。他ツール向けの安定した interchange schema ではなく、Qubex /
  Python の永続化ファイルとして扱ってください。
- 信頼できる `ExperimentRecord` ファイルだけを読み込んでください。jsonpickle
  の decode は Python object を復元できます。
- 古い record を復元できるかどうかは、保存時と互換性のある Python package と
  import path があるかに依存します。

## 特定の実験処理だけ raw `MeasurementResult` を保存する

`MeasurementResult` は、1 回の測定実行を表す Qubex 標準の測定結果 model です。
target ごとの raw data、`MeasurementConfig`、任意の device configuration、
classifier reference を持ちます。

特定の実験処理だけ raw data と metadata を残したい場合は `save_rawdata()` を
使います。

```python
with exp.system_manager.save_rawdata(rawdata_dir=".rawdata", tag="q00-rabi"):
    analyzed = exp.obtain_rabi_params(targets=["Q00"], n_shots=1024)
```

この context の中で実行された測定ごとに、`MeasurementResult` が自動保存されます。
ファイルは `.rawdata/q00-rabi/YYYYMMDD_HHMMSS_microseconds.nc` として書かれます。

保存された raw file は model loader で読み戻します。

```python
from qubex.measurement.models import MeasurementResult

raw_result = MeasurementResult.load(".rawdata/q00-rabi/20260604_101530_123456.nc")
```

context manager を抜けると、例外が発生した場合も含めて、以前の raw-data 設定へ
戻ります。

## `execute()` / `measure()` の raw-data 保存を有効にし続ける

notebook や実験セッション中に `execute()` または `measure()` を繰り返し実行し、
それらの raw data を保存したい場合は、最初に raw-data directory を設定し、
セッション終了時に明示的に無効化します。

```python
exp.system_manager.set_rawdata_dir(".rawdata/20260604-q00-session")

try:
    result_a = exp.execute(schedule_a, n_shots=1024)
    result_b = exp.measure(sequence_b, n_shots=1024)
finally:
    exp.system_manager.set_rawdata_dir(None)
```

!!! note
    多くの `execute()` / `measure()` 呼び出しで raw-data 保存を有効にしたままに
    すると、測定条件によっては保存容量が大きくなることがあります。長時間の実験や
    共有サーバーで使う場合は、保存先の空き容量とセッションごとの整理に注意して
    ください。

`SystemManager` の状態は active な実験 stack から共有されることがあるため、
raw-data 保存を複数の呼び出しにまたがって有効にするときは `try` / `finally` を
使うことを推奨します。

この設定が効くのは、`execute()` や `measure()` のように自動 raw-data 保存の経路を
通る API です。API から `MeasurementResult` を直接受け取る場合は、次のように返り値
を明示的に保存してください。

## 返された `MeasurementResult` を明示的に保存する

API から `MeasurementResult` を直接受け取った場合は、返り値を明示的に保存します。

```python
from qubex.measurement.models import MeasurementResult

raw_result = await exp.run_measurement(
    schedule=measurement_schedule,
    n_shots=1024,
)

path = raw_result.save("results/q00-run-measurement.nc")
restored = MeasurementResult.load(path)
```

timestamp 付きの自動 raw-data file ではなく、自分で filename を決めたい場合にも
この方法が使えます。

## `MeasurementResult` の構造

論理構造は次の通りです。

- `MeasurementResult.data`: target label を key にした
  `dict[str, list[CaptureData]]`
- `MeasurementResult.measurement_config`: shot 数、shot interval、averaging
  mode、integration mode、classification mode、要求した return item
- `MeasurementResult.device_config`: 任意の device / backend snapshot
- `MeasurementResult.classifier_refs`: target ごとの任意の classifier metadata

各 `CaptureData` entry は次の情報を持ちます。

- `target`
- `config`
- `payload`
- `sampling_period`
- `classifier_ref`

primary capture array は `config.primary_return_item` で選ばれ、`capture.data` と
して参照できます。payload field は次の標準 shape を使います。

| Payload field | Shape |
| --- | --- |
| `waveform_series` | `(n_shots, capture_length)` |
| `iq_series` | `(n_shots,)` |
| `state_series` | `(n_shots,)` |
| `averaged_waveform` | `(capture_length,)` |
| `averaged_iq` | scalar |

## NetCDF4 file contract

`MeasurementResult.save()` は `save_netcdf()` の alias です。ファイルは NetCDF4
として書かれ、Qubex/qxcore の metadata を持ちます。

- global attribute `format`
- global attribute `format_version`
- global attribute `model_class`
- global attribute `payload_json`

NetCDF4 を使うためファイルは HDF5 ベースですが、公開 contract は手書きの HDF5
group layout ではなく、Qubex `DataModel` loader です。top-level の配列は
NetCDF variable として保存されることがあります。一方、nested model payload は
`payload_json` に encode されることがあります。安定して復元するには、内部の
variable name に依存せず `MeasurementResult.load()` を使ってください。

手動で `netCDF4` から確認する場合は、complex support を有効にして開きます。

```python
from netCDF4 import Dataset

with Dataset("result.nc", mode="r", auto_complex=True) as ds:
    print(ds.ncattrs())
```
