# インストール

Qubex は Python 3.10 以上で動作します。ハードウェアバックエンドを使わない範囲では OS を問いませんが、ハードウェアバックエンドを使う場合は追加依存関係が必要で、通常は Linux ホストを前提にします。
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

### QuEL-1 バックエンド付きでインストールする（Linux）

Linux ホスト上で Qubex を実機と組み合わせて使う予定なら、この構成を使ってください。
`QuantumSimulator` だけを使う場合は、`backend` / `quel1` extra は不要です。

```bash
uv pip install "qubex[backend] @ git+https://github.com/amachino/qubex.git"
```

これにより、公開されている QuEL-1 向けバックエンドライブラリがインストールされます。
`qubex[quel1]` も同等の explicit な extra 名として使えます。

### QuEL-3 向けにインストールする

QuEL-3 の runtime support を使う予定なら、この構成を使ってください。

```bash
uv pip install "qubex[quel3] @ git+https://github.com/amachino/qubex.git"
```

`quel3` は `quelware-client` に依存します。これが PyPI に公開されていない
場合は、利用する環境で指定された package source や index から追加で
インストールしてください。

### 特定バージョンをインストールする

再現性のためにバージョンを固定したい場合はこちらを使ってください。

```bash
uv pip install "qubex[backend] @ git+https://github.com/amachino/qubex.git@<version>"
```

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
