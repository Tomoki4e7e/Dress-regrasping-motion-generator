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

---

# 具体的な計算過程（実装コード抜粋）

以下は `data-processing/` の実装から、特徴量が計算される順序に沿って主要部分を抜粋したもの。コードが仕様の正本であり、各抜粋のリンクは実装ファイルを示す。

## 1. マスク・深度画像の読込とマスク適用

`EpisodeReader` は画像をグレースケール `uint8` で読み、既定ではオンライン処理と同じ `[:960, :1280]` に crop する。マスク適用深度は、マスク値が `255` の画素だけ元の深度値を残す。

出典: [`shareset_io/reader.py`](shareset_io/reader.py)

```python
def apply_mask_to_depth(
    mask: np.ndarray,
    depth: np.ndarray,
    foreground: int = MASK_FOREGROUND,
) -> np.ndarray:
    if mask.shape != depth.shape:
        raise ValueError(f"mask/depth shape mismatch: {mask.shape} vs {depth.shape}")
    return np.where(mask == foreground, depth, 0).astype(np.uint8)


def crop_to_contract(image: np.ndarray) -> np.ndarray:
    return image[:CROP_HEIGHT, :CROP_WIDTH]
```

フレーム番号は PNG ファイル名の stem を整数として昇順に並べる。したがって `10.png` が `2.png` より先になる辞書順ソートは使用しない。

```python
return sorted((p.name for p in names), key=lambda x: int(Path(x).stem))
```

## 2. 被覆率の計算

### 2.1 足領域面積

各フレームの `leg_mask` について、前景値 `255` の画素数を足の可視面積 \(A_t\) とする。

出典: [`coverage/compute.py`](coverage/compute.py)

```python
def foreground_area(mask: np.ndarray, foreground: int = MASK_FOREGROUND) -> int:
    return int(np.count_nonzero(mask == foreground))
```

`leg_mask` が欠落したフレームは面積を `NaN` とし、`missing_frames` にフレーム番号を記録する。

```python
frame = reader.load_frame(frame_id, require=("leg_mask",))
if frame.leg_mask is None or "leg_mask" in frame.missing:
    areas.append(float("nan"))
    missing.append(frame_id)
    continue
areas.append(float(foreground_area(frame.leg_mask)))
```

### 2.2 baseline と 0〜1 正規化

既定の `early_max` では、先頭 `baseline_window` フレーム（既定 30）の有限な面積の最大値を baseline \(A_0\) とする。先頭区間に有効値がなければ、エピソード全体の最大値へフォールバックする。

```python
early = leg_areas[:baseline_window]
early_valid = early[np.isfinite(early)]
if early_valid.size == 0:
    return float(np.max(valid))
return float(np.max(early_valid))
```

被覆率は可視足領域の減少率として計算し、`[0, 1]` に clip する。

\[
\mathrm{coverage}_t =
\mathrm{clip}\left(\frac{A_0-A_t}{A_0},\,0,\,1\right)
\]

```python
areas = np.asarray(leg_areas, dtype=np.float64)
out = np.full(areas.shape, np.nan, dtype=np.float64)
if not np.isfinite(baseline_area) or baseline_area <= 0:
    return out
valid = np.isfinite(areas)
raw = (baseline_area - areas[valid]) / baseline_area
out[valid] = np.clip(raw, 0.0, 1.0)
```

最終的な `coverage_ok` は、計算結果が有限値かどうかで決まる。

```python
coverage = coverage_from_areas(leg_area, baseline)
ok = np.isfinite(coverage)
```

## 3. 靴下開口縁 N 点の計算

### 3.1 二値化と最大外輪郭

`sock_mask` と `leg_mask` は `255` の画素だけを 1 とする二値画像へ変換する。靴下について `cv2.RETR_EXTERNAL` で外輪郭を取得し、面積最大の輪郭を採用する。

出典: [`keypoints/compute.py`](keypoints/compute.py)

```python
def _binarize(mask: np.ndarray) -> np.ndarray:
    return (mask == MASK_FOREGROUND).astype(np.uint8)


def largest_external_contour(mask_bin: np.ndarray):
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)
```

### 3.2 足に近い連続弧の抽出

`leg_mask` の距離変換を作り、靴下外輪郭上の各点から足領域までの距離を求める。距離が下位 20 パーセンタイル以内（ただし閾値は最低 2 px）の点を「足に近い点」とする。

```python
leg_inv = np.where(leg_bin > 0, 0, 255).astype(np.uint8)
dist = cv2.distanceTransform(leg_inv, cv2.DIST_L2, 5)
xs = np.clip(pts[:, 0].astype(np.int32), 0, dist.shape[1] - 1)
ys = np.clip(pts[:, 1].astype(np.int32), 0, dist.shape[0] - 1)
dvals = dist[ys, xs]
thr = float(np.percentile(dvals, percentile))
near = dvals <= max(thr, 2.0)
```

輪郭は閉曲線なので `near` を2回連結し、`True` が最長となる連続区間を探索する。この区間を開口縁候補の弧とする。

```python
near_ext = np.concatenate([near, near])
best_len, best_start, cur, start = 0, 0, 0, 0
for i, flag in enumerate(near_ext):
    if flag:
        if cur == 0:
            start = i
        cur += 1
        if cur > best_len:
            best_len, best_start = cur, start
    else:
        cur = 0
```

### 3.3 弧長等間隔サンプリング

開口縁候補を構成する隣接点間のユークリッド距離を累積し、弧全体を `N` 等分した位置へ線形補間する。既定値は `N=8`。

```python
seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
cum = np.concatenate([[0.0], np.cumsum(seg)])
total = float(cum[-1])
targets = np.linspace(0.0, total, n)

for i, t in enumerate(targets):
    j = int(np.searchsorted(cum, t, side="right") - 1)
    j = min(max(j, 0), len(seg) - 1)
    if seg[j] < 1e-9:
        out[i] = pts[j]
    else:
        a = (t - cum[j]) / seg[j]
        out[i] = pts[j] * (1.0 - a) + pts[j + 1] * a
```

## 4. つま先・踵の計算

靴下と足の二値領域の和集合を点群 \((x,y)\) に変換し、その共分散行列を固有値分解する。最大固有値に対応する固有ベクトルを足全体の主軸とする。

```python
union = ((sock_bin > 0) | (leg_bin > 0)).astype(np.uint8)
uy, ux = np.nonzero(union)
pts = np.column_stack([ux, uy]).astype(np.float64)
mean = pts.mean(axis=0)
cov = np.cov(pts.T)
eigvals, eigvecs = np.linalg.eigh(cov)
axis = eigvecs[:, int(np.argmax(eigvals))]
```

固有ベクトルの符号は任意なので、靴下重心側が遠位方向、足重心側が近位方向になるように向きを揃える。

```python
sock_c = np.array([sx.mean(), sy.mean()], dtype=np.float64)
leg_c = np.array([lx.mean(), ly.mean()], dtype=np.float64)
if (sock_c - mean) @ axis < (leg_c - mean) @ axis:
    axis = -axis
```

各マスク画素を主軸へ射影し、靴下領域の最大射影点をつま先、足領域の最小射影点を踵とする。

```python
mp = np.column_stack([mx, my]).astype(np.float64)
proj = (mp - mean) @ axis
idx = int(np.argmax(proj) if distal else np.argmin(proj))
return mp[idx]

toe = extreme(sock_bin, distal=True)
heel = extreme(leg_bin, distal=False)
```

## 5. 深度付与と座標正規化

### 5.1 深度値

各キーポイントの \((x,y)\) を最近傍整数画素へ丸め、同じ画素の `camera_depth[y, x]` を取得する。深度画像の欠落、非有限座標、画像範囲外は `NaN` となる。現在の実装では画素値 0 は `NaN` へ変換せず、そのまま `0.0` として返す。

```python
if depth is None or not np.all(np.isfinite(xy)):
    return float("nan")
x = int(round(float(xy[0])))
y = int(round(float(xy[1])))
h, w = depth.shape[:2]
if x < 0 or y < 0 or x >= w or y >= h:
    return float("nan")
return float(depth[y, x])
```

### 5.2 つま先原点・足長正規化

踵とつま先のユークリッド距離を足長 \(L\) とし、開口縁、踵、つま先をすべて同じ式で正規化する。

\[
L = \lVert \mathrm{heel}-\mathrm{toe}\rVert_2,\qquad
p' = \frac{p-\mathrm{toe}}{L}
\]

```python
foot_length = float(np.linalg.norm(heel - toe))
if not np.isfinite(foot_length) or foot_length < MIN_FOOT_LENGTH:
    return np.full_like(points, np.nan, dtype=np.float64), float("nan")
return (points - toe) / foot_length, foot_length
```

計算対象を一度縦に連結してから同じ足長で正規化し、出力配列へ分割する。このため `toe_norm` はほぼ `(0, 0)`、`heel_norm` のノルムはほぼ 1 になる。

```python
stacked = np.vstack([opening, heel.reshape(1, 2), toe.reshape(1, 2)])
stacked_norm, foot_length = normalize_about_toe(stacked, toe, heel)

opening_norm = stacked_norm[:n_opening]
heel_norm = stacked_norm[n_opening]
toe_norm = stacked_norm[n_opening + 1]
```

## 6. フレーム失敗処理と学習用統合

キーポイント計算に失敗したフレームは、全特徴量を `NaN`、`ok=False` とする。主な失敗理由はマスク欠落、空領域、短すぎる開口弧、足長不足である。

```python
return FrameKeypoints(
    frame_id=frame_id,
    opening_xy=np.full((n_opening, 2), np.nan, dtype=np.float64),
    opening_depth=np.full((n_opening,), np.nan, dtype=np.float64),
    opening_norm=np.full((n_opening, 2), np.nan, dtype=np.float64),
    heel_xy=nan2.copy(),
    toe_xy=nan2.copy(),
    heel_depth=float("nan"),
    toe_depth=float("nan"),
    heel_norm=nan2.copy(),
    toe_norm=nan2.copy(),
    foot_length=float("nan"),
    ok=False,
    fail_reason=reason,
)
```

Step 4 では coverage と keypoints を同じ `start:end` で計算し、両方が有効なフレームだけを `features_ok=True` とする。

出典: [`feature_package/compute.py`](feature_package/compute.py)

```python
coverage = compute_episode_coverage(
    reader,
    start=start,
    end=end,
    baseline_mode=baseline_mode,
    baseline_window=baseline_window,
    fixed_baseline=fixed_baseline,
)
keypoints = compute_episode_keypoints(
    reader,
    start=start,
    end=end,
    n_opening=n_opening,
)
coverage_ok = coverage.ok.astype(bool)
keypoints_ok = keypoints.ok.astype(bool)
features_ok = coverage_ok & keypoints_ok
```

したがって学習・報酬計算では `features_ok` を有効フレームマスクとして使用できる。無効フレームの補間や代入は前処理では行わず、下流側で明示的に決める。
