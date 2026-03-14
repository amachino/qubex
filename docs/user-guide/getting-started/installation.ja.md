# インストール

Qubex は Python 3.10 以上で動作し、macOS と Linux をサポートします。ハードウェアバックエンドを使う場合は追加依存関係が必要で、通常は Linux ホストを前提にします。
`v1.5.0 beta` 期間中は、正式リリースまで Qubex のインストールに `uv` が必要です。
正式リリースでは Qubex を PyPI に公開し、`pip install` を使えるようにする予定です。

## Python 環境を準備する

Python 本体と仮想環境の管理には [uv](https://docs.astral.sh/uv/) を推奨します。`uv` は高速なスタンドアロンツールで、再現性のある環境を維持しやすくなります。

まだ Python を用意していない場合は、先にインストールしてください。

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
uv pip install "qubex @ git+https://github.com/amachino/qubex.git@main"
```

### 実機利用向けのバックエンド付きでインストールする（Linux）

Linux ホスト上で Qubex を実機と組み合わせて使う予定なら、この構成を使ってください。
Simulator だけを使う場合は、`backend` extra は不要です。

```bash
uv pip install "qubex[backend] @ git+https://github.com/amachino/qubex.git@main"
```

これにより、実機実行に必要なバックエンドライブラリがインストールされます。
QuEL-3 を使う場合は、互換性のある `quelware-client` のソースまたはパッケージも必要です。
リポジトリ開発時には `packages/quelware-client` が submodule として取得され、ローカルに `lib/quelware-client-internal` が存在する場合はそちらが優先されます。

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

- 実機を使うなら、次に [システム設定](system-configuration.md) を進めてください。
- 自分の目的に合うワークフローは [入口を選ぶ](choose-your-entry-point.md) から確認できます。
