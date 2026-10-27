# インストール

Qubex は Python 3.10 以上で動作します。QuEL-1 を使う場合のみ Linux が必要です。
`v1.5.0 beta` 期間中は、正式リリースまで Qubex のインストールに `uv` が必要です。
正式リリースでは Qubex を PyPI に公開し、`pip install` を使えるようにする予定です。

## Python 環境を準備する

Python 本体と仮想環境の管理には [uv](https://docs.astral.sh/uv/) を推奨します。`uv` は Python のインストール、仮想環境の作成、依存関係の管理を一貫して扱えるツールです。

## 仮想環境を作成する

システムパッケージや他のプロジェクトとの衝突を避けるため、専用の仮想環境に Qubex をインストールすることを推奨します。

プロジェクトディレクトリで次のいずれかを実行してください。

=== "uv"
    ```bash
    uv venv
    ```

=== "venv"
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

## Qubex をインストールする

このセクションでは、標準利用、バックエンド付き構成、バージョン固定、開発用途のインストール方法を紹介します。

### 最新版をインストールする

リポジトリ上の最新機能を使いたい場合はこちらを使ってください。

```bash
uv pip install "qubex @ git+https://github.com/amachino/qubex.git"
```

### QuEL-1・QuEL-3 対応でインストールする（Linux 限定）

`backend` を指定すると、QuEL-1 と QuEL-3 の両方の依存パッケージがインストールされます。

```bash
uv pip install "qubex[backend] @ git+https://github.com/amachino/qubex.git"
```

シミュレータのみを使う場合、ハードウェア向けの extra は不要です。

### QuEL-1 のみ対応でインストールする（Linux 限定）

```bash
uv pip install "qubex[quel1] @ git+https://github.com/amachino/qubex.git"
```

### QuEL-3 のみ対応でインストールする

```bash
uv pip install "qubex[quel3] @ git+https://github.com/amachino/qubex.git"
```

### 特定バージョンをインストールする

再現性のためにバージョンを固定したい場合はこちらを使ってください。

```bash
uv pip install "qubex @ git+https://github.com/amachino/qubex.git@<version>"
```

実機向けの依存パッケージも導入する場合は、上記の例の `qubex` を `qubex[backend]`、
`qubex[quel1]`、または `qubex[quel3]` に置き換えてください。

### 開発用にインストールする

Qubex 本体を開発・テストするためのセットアップです。
このリポジトリの開発コマンドは `uv` 管理環境を前提にしています。

```bash
git clone -b develop https://github.com/amachino/qubex.git
cd qubex
make sync
```

## 次のステップ

- まず [始め方を選ぶ](choose-where-to-start.md) で、自分の目的に合う入口を確認してください。
- 実機を使う `Experiment` または `低レベル API` に進む場合は、その後で [システム設定](system-configuration.md) を用意してください。
