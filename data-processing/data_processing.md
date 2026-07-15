# 開発フロー
## 従来の研究
    - 画像の靴下、足のセグメンテーション
    - 深度カメラから靴下の深度画像の生成
    - 靴下、足のマスク画像、靴下、足の深度画像のマスク
    - SegmentAnythingModel2, DepthAnythingModel2を使用
        - ShareSetディレクトリを参照
## 新規開発
    - 足及び靴下が写っている画像から以下の位置を検出する
        - 靴下の開口部の縁の周囲8点
        - 足の踵、つま先
            - 画像上の点として検出
            - 深度画像から奥行き情報を得たい
            - つま先を原点として他の点を正規化する
    - 靴下領域、足領域から被覆率を求める
        - 各画像の被覆率を0.0 ~ 1.0で正規化

---

# 開発記録（Step 0〜4）

ShareSet は読取専用。`data-processing/` はこの上に新規特徴だけを追加する。

| Step | モジュール | 内容 |
|------|------------|------|
| 0 | ShareSet 契約 | マスク／深度の入出力形 |
| 1 | `shareset_io/` | 読込ラッパ |
| 2 | `coverage/` | 被覆率 |
| 3 | `keypoints/` | 開口縁・踵/つま先・正規化 |
| 4 | `feature_package/` | 学習用パッケージ統合 |

詳細ルールの正本: `.cursor/rules/data_processing_step0.mdc`〜`step4.mdc`

## Step 0 — 入出力契約（確定）

### エピソード構成

```
episode/
  camera_right/          # RGB
  camera_right_mask/
    sock_mask/
    leg_mask/
  camera_depth/
  depth_mask/
    sock_depth/
    leg_depth/
  angle.csv / torque.csv / touch.csv
```

参照データ例: `data_center_general_depth_n5_sock`、`dataset_sock.yaml`

### 共通コア演算

```
where(mask == 255, depth, 0)
```

- 前景: mask `255`、背景: `0`
- 対象: `sock` / `leg`（オンラインでは foot = leg）
- 解像度契約: **1280×960 crop**

### オフライン変換 — `mask_convert_to_depth.py`

| 項目 | 形 |
|------|-----|
| 入力 mask | `camera_right_mask/{sock,leg}_mask/*.png`、`(H,W)` uint8 |
| 入力 depth | `camera_depth/*.png`、`(H,W)` uint8 |
| 演算 | `where(mask==255, depth, 0)` |
| 出力 | `depth_mask/{sock,leg}_depth/{i}.png`、同形状 uint8 |

ファイルは stem 整数順（`0.png`, `1.png`, …）。フレームは zip 対応。

### 学習用読み込み — `MaskDepthDataLoader`

| 項目 | 形 |
|------|-----|
| パス | `depth_mask/{sock,leg}_depth/{i}.png` |
| 前処理 | GaussianBlur(15) → crop(1280×960) → resize(`img_size`) |
| テンソル | `(T, 1, H, W)` → 0–255 を `[vmin, vmax]` |
| バッチ | `(N, T, 1, H, W)` |

`MaskDataLoader` はパスだけ `camera_right_mask/{sock,leg}_mask/` で同型。

### オンライン — `eipl_online_control_SAMDAMSARNN_gripper_sock.py`

| 項目 | 形 |
|------|-----|
| crop | `[:960, :1280]` |
| mask / depth | SAM2 / DepthAnything → uint8、前景 255 |
| 演算 | `where(mask==255, depth, 0)`（sock / foot） |
| モデル入力 | `[1, 1, H, W]` |

### 横断まとめ

| | オフライン | 学習ローダ | オンライン |
|--|-----------|------------|------------|
| 演算 | `where(...)` | 保存済み depth_mask | 同じ `where` |
| 対象 | sock / leg | sock / leg | sock / foot(=leg) |
| 深度源 | `camera_depth` | `depth_mask/*_depth` | DepthAnything + `depth.json` |
| 出力 | `(H,W)` PNG | `(T,1,H,W)` | `[1,1,H,W]` |

## Step 1 — 既存出力の読み込みラッパ（確定）

被覆率・キーポイントは必ずこのラッパ経由でマスク／深度を取る。

### 配置

```
dress_regrasping/data-processing/shareset_io/
  __init__.py
  paths.py      # エピソード相対パス・crop 定数
  reader.py     # EpisodeReader / FrameBundle / apply_mask_to_depth
```

### 公開 API

```python
from shareset_io import EpisodeReader, apply_mask_to_depth, MASK_FOREGROUND

reader = EpisodeReader("/path/to/episode")
frame = reader.load_frame(0)          # FrameBundle
leg = reader.load_mask("leg", 0)      # (H,W) uint8
sock = reader.load_mask("sock", 0)
depth = reader.load_camera_depth(0)
sock_d = reader.load_depth_mask("sock", 0)
recomputed = reader.compute_depth_mask("sock", 0)

for frame in reader.iter_frames(start=10, end=252):
    ...
```

`PYTHONPATH` に `dress_regrasping/data-processing` を足すか、そのディレクトリを CWD にする。

### 契約（Step 0 準拠）

| 項目 | 内容 |
|------|------|
| パス | `camera_right_mask/{sock,leg}_mask`、`camera_depth`、`depth_mask/{sock,leg}_depth` |
| 形 | `(H, W)` uint8 grayscale（crop 後は最大 960×1280） |
| 前景 | mask `255`、背景 `0` |
| コア演算 | `apply_mask_to_depth` → `where(mask == 255, depth, 0)` |
| ソート | stem 整数順（`0.png`, `1.png`, …） |
| `foot` | `leg` の別名 |

### このラッパがやらないこと

- GaussianBlur / resize / `[vmin,vmax]` 正規化は行わない
- セグメンテーション・深度推定の再実行はしない
- 被覆率・キーポイント計算は Step 2 以降

### 失敗フラグ

`FrameBundle.missing` に欠落キーを入れる。`FrameBundle.ok` は欠落なし。下流は欠損フレームを明示扱いする。

## Step 2 — 被覆率（確定）

入力は `EpisodeReader` 経由のフル解像度 `leg_mask`。

### 定義

```
A0 = baseline（既定: 先頭 baseline_window フレームの leg 前景画素数の最大）
coverage_t = clip((A0 - A_t) / A0, 0, 1)
```

- 前景: mask `== 255`（`MASK_FOREGROUND`）
- 範囲: **0.0〜1.0**
- `leg_mask` 欠落: coverage / area は NaN、`ok=false`、`missing_frames` に記録

| mode | 意味 |
|------|------|
| `early_max`（既定） | 先頭 `baseline_window`（既定 30）の max `A` |
| `episode_max` | 全有効フレームの max `A` |
| `fixed` | 外部指定の `A0` |

### 配置

```
dress_regrasping/data-processing/coverage/
  __init__.py
  compute.py
  __main__.py
```

### 公開 API

```python
from shareset_io import EpisodeReader
from coverage import compute_episode_coverage, save_episode_coverage

reader = EpisodeReader("/path/to/episode")
result = compute_episode_coverage(reader)
save_episode_coverage(result, reader.episode_dir / "features", episode_dir=reader.episode_dir)
```

### 出力（`features/`）

| ファイル | 内容 |
|----------|------|
| `coverage.npy` | `(T,)` float32、欠落は NaN |
| `leg_area.npy` | `(T,)` float64、前景画素数 |
| `features.json` | coverage / leg_area / ok / missing_frames / baseline メタ |

### CLI

```bash
cd dress_regrasping/data-processing
PYTHONPATH=. python3 -m coverage \
  --dataset-split ../../ShareSet/share_folder/data/data_center_general_depth_n5_sample/train
```

## Step 3 — キーポイント（確定）

入力は `EpisodeReader` 経由の `sock_mask` / `leg_mask` / `camera_depth`。

### 定義

| 対象 | 方針 |
|------|------|
| 開口縁 `N` 点 | `sock_mask` 最大外輪郭のうち、`leg_mask` に最も近い連続弧を弧長等間隔サンプリング |
| 主軸 | `sock ∪ leg` 前景の PCA。極性は sock 重心が遠位側 |
| つま先 | sock 上の遠位端点 |
| 踵 | leg 上の近位端点（上端 crop 張り付きは許容） |
| 奥行き | 同一画素の `camera_depth`（欠落・OOB は NaN） |

正規化（足長 `L = ||heel - toe||`）:

```
p' = (p - toe) / L
```

- `toe_norm ≈ (0, 0)`、`heel_norm` は単位方向
- `L < 1` px は失敗（`foot_length_zero`）
- 縁点数 `N`: 仕様上 **8**（API/CLI で **6〜8**）

### 配置

```
dress_regrasping/data-processing/keypoints/
  __init__.py
  compute.py
  __main__.py
```

### 公開 API

```python
from shareset_io import EpisodeReader
from keypoints import compute_episode_keypoints, save_episode_keypoints

reader = EpisodeReader("/path/to/episode")
result = compute_episode_keypoints(reader, n_opening=8)
save_episode_keypoints(result, reader.episode_dir / "features", episode_dir=reader.episode_dir)
```

### 出力（`features/`）

| ファイル | 内容 |
|----------|------|
| `opening_xy.npy` | `(T, N, 2)` float32、欠落 NaN |
| `opening_depth.npy` | `(T, N)` float32 |
| `opening_norm.npy` | `(T, N, 2)` float32 |
| `heel_xy.npy` / `toe_xy.npy` | `(T, 2)` |
| `heel_depth.npy` / `toe_depth.npy` | `(T,)` |
| `heel_norm.npy` / `toe_norm.npy` | `(T, 2)` |
| `foot_length.npy` | `(T,)` |
| `keypoints_ok.npy` | `(T,)` bool |
| `features.json` | 既存 coverage を消さず `keypoints` ブロックを merge |

失敗例: `missing_sock_mask` / `empty_sock` / `opening_too_short` / `foot_length_zero`。埋め合わせは下流ポリシー側。

### CLI

```bash
cd dress_regrasping/data-processing
PYTHONPATH=. python3 -m keypoints \
  --dataset-split ../../ShareSet/share_folder/data/data_center_general_depth_n5_sample/train \
  --n-opening 8
```

## Step 4 — 学習用パッケージ出力（確定）

入力は Step 2/3 の結果をエピソード配下 `features/` にまとめる。

### 目的

縁点・踵/つま先（2D＋深度）、正規化座標、被覆率、失敗フラグを学習入力と報酬で同一定義で使えるパッケージにする。補間・埋め込みはしない。

### 配置

```
dress_regrasping/data-processing/feature_package/
  __init__.py
  compute.py
  __main__.py
```

### 公開 API

```python
from shareset_io import EpisodeReader
from feature_package import (
    compute_episode_features,
    save_episode_features,
    load_episode_features,
)

reader = EpisodeReader("/path/to/episode")
bundle = compute_episode_features(reader, n_opening=8)
save_episode_features(bundle, reader.episode_dir / "features", episode_dir=reader.episode_dir)
loaded = load_episode_features(reader.episode_dir / "features")
```

### 出力（`features/`）

Step 2/3 の npy は維持。追加:

| ファイル | 内容 |
|----------|------|
| `coverage_ok.npy` | `(T,)` bool |
| `features_ok.npy` | `(T,)` bool = `coverage_ok & keypoints_ok` |
| `features.json` の `package` | 統合メタ・失敗要約 |

参照: `coverage.npy` / `leg_area.npy`（Step 2）、`opening_*` / `heel_*` / `toe_*` / `foot_length.npy` / `keypoints_ok.npy`（Step 3）。

### 失敗ポリシー

- 欠落値は **NaN**（bool は False）
- **埋め合わせはしない**
- `features_ok[t]==False` は学習・報酬でスキップまたはマスクする前提

### CLI

```bash
cd dress_regrasping/data-processing
PYTHONPATH=. python3 -m feature_package \
  --dataset-split ../../ShareSet/share_folder/data/data_center_general_depth_n5_sample/train \
  --n-opening 8
```
