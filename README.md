# 「LightGBM + AD（DoE）」 vs 「GPR + Acquisition function（BO）」
![Pipeline Comparison](./images/pipeline_comparison_gemini.png)

## 要約

本リポジトリは、[Deep4Chem](http://deep4chem.korea.ac.kr/) 由来の有機発光材料データセットを対象に、**長波長発光（Emission max）** と **高量子収率（Quantum yield / PLQY）** を同時に満たす候補化合物を探索するための再現可能なノートブック群を提供する。

[金子弘昌『Pythonで学ぶ実験計画法入門 — ベイズ最適化によるデータ解析』](https://www.amazon.co.jp/%EF%BC%B0%EF%BD%99%EF%BD%84%EF%BD%88%EF%BD%8F%EF%BD%8E%E3%81%A7%E5%AD%A6%E3%81%B6%E5%AE%9F%E9%A8%93%E8%A8%88%E7%94%BB%E6%B3%95%E5%85%A5%E9%96%80-%E3%83%99%E3%82%A4%E3%82%BA%E6%9C%80%E9%81%A9%E5%8C%96%E3%81%AB%E3%82%88%E3%82%8B%E3%83%87%E3%83%BC%E3%82%BF%E8%A7%A3%E6%9E%90-%EF%BC%AB%EF%BC%B3%E6%83%85%E5%A0%B1%E7%A7%91%E5%AD%A6%E5%B0%82%E9%96%80%E6%9B%B8-%E9%87%91%E5%AD%90%E5%BC%98%E6%98%8C-ebook/dp/B09C89HZRV)（以下、*python_doe_kspub* 系）のワークフローをベースに、Deep4Chem データへ適用したものである。

パイプラインは大きく 2 系統に分かれる。

1. **DoE 系（LightGBM + 適用域 AD）** — 学習済み LightGBM でテスト候補を一括予測し、D 最適基準・OCSVM-AD・閾値フィルタで絞り込む（`5.Predict_test_LGB.ipynb`）
2. **ベイズ最適化系（GPR + 獲得関数）** — ガウス過程回帰（GPR）の不確かさを利用し、複数目的を同時に考慮しながら候補を逐次選抜する（`7`, `8` 番ノートブック）

※広義の意味では、ベイズ最適化による材料の探索もDoE（Design of Experiment）かもしれないが、ここでは区別するために「LightGBM + AD」をDoE、「GPR + 獲得関数」をBOとする。実務ではどちらの手法でも、予測後に良い結果を示すと判断された材料を新たに合成し、その結果をデータセットに加えてモデルを更新（再学習）し、さらに新しい候補を次の実験候補にしていく。

## 背景
### 探索課題

有機発光材料において、長波長でよく光る材料は需要がある。例えば、顔認証などに使われるsensingの技術では効率的な近赤外発光が必要である。ただこれらの長波長を示す材料は「**Band gap law**」により量子収率が低い問題がある。そこで、本プロジェクトでは次の2つの目的変数を最大化対象とする。

| 目的変数 | 単位 | 方向 |
|---|---|---|
| Emission max (nm) | nm | 大きいほど良い |
| Quantum yield (PLQY) | 0–1 | 大きいほど良い |

実験可能な候補数に比べ化学空間は広いため、**少ない実験回数で有望な候補を効率よく選ぶ**ことが重要である。

### ベイズ最適化（Bayesian Optimization, BO）

ベイズ最適化は、評価コストの高いブラックボックス関数（ここでは「記述子 → 物性」）に対し、以下の 2 要素を組み合わせて次に実験する点を選ぶ手法である。

1. **サロゲートモデル（代理モデル）** — 既知データから目的変数を予測し、**予測の不確かさ**も推定する。本リポジトリでは主に **GPR（ガウス過程回帰）** を用いる。
2. **獲得関数（acquisition function）** — サロゲートの予測平均と不確かさから、「改善が期待できる」候補をスコア化する。

BO の利点は、不確かさが大きい未探索領域（**探索**）と、現時点で有望な領域（**活用**）のバランスを取りながら候補を選べる点にある。本リポジトリでは多目的設定に拡張し、発光波長・PLQY を同時に扱う。

| ノートブック | 獲得関数の考え方 |
|---|---|
| `7.BO_multi_y_multi_sample.ipynb` | 各目的の目標達成確率（PI / PTR）の **log 確率和** を最大化（金子先生書籍のやり方） |
| `8.BO_multi_y_multi_sample.ipynb` | **constrained EI**（PLQY の EI × 発光波長範囲内確率）または **weighted UCB** |


### 獲得関数の定義（ノートブック 7・8 の実装）

候補 $x$ に対し、各目的変数ごとに GPR が予測平均 $\mu(x)$ と標準偏差 $\sigma(x)$ を返す。$\Phi$, $\phi$ はそれぞれ標準正規分布の累積分布関数・密度関数とする。実装では `scipy.stats.norm` を用い、$\texttt{relaxation} = \varepsilon = 0.01$ を既定とする。

#### 共通：目標達成確率（ノートブック 7）

**PI（Probability of Improvement, 最大化）** — 学習データの現状最良値を上回る確率

$$
f^{*} = \max_{i \in \text{train}} y_i + \varepsilon \,\mathrm{std}(y_{\text{train}}), \qquad
P_{\mathrm{PI}}(x) = P\bigl(Y(x) > f^{*}\bigr) = 1 - \Phi\!\left(\frac{f^{*} - \mu(x)}{\sigma(x)}\right)
$$

- $f^*$: 改善判定に使う閾値（best-so-far に緩和項を加えた値）
- $\varepsilon$: 緩和係数（既定 0.01）
- $\Phi$: 標準正規分布の累積分布関数（CDF）
- $\mu(x), \sigma(x)$: GPR の予測平均・予測標準偏差

---

**PI（最小化）** — `settings` で `target_type = -1` のとき

$$
f^{*} = \min_{i \in \text{train}} y_i - \varepsilon \,\mathrm{std}(y_{\text{train}}), \qquad
P_{\mathrm{PI}}(x) = P\bigl(Y(x) < f^{*}\bigr) = \Phi\!\left(\frac{f^{*} - \mu(x)}{\sigma(x)}\right)
$$

- $f^*$: 改善判定に使う閾値（best-so-far に緩和項を引いた値）
- $\varepsilon$: 緩和係数（既定 0.01）
- $\Phi$: 標準正規分布の累積分布関数（CDF）
- $\mu(x), \sigma(x)$: GPR の予測平均・予測標準偏差

---

**PTR（Probability of Target Range, 範囲指定）** — 予測値が目標区間 $[y_{\mathrm{low}}, y_{\mathrm{high}}]$ に入る確率

$$
\begin{aligned}
P_{\mathrm{PTR}}(x) &= P\bigl(y_{\mathrm{low}} \le Y(x) \le y_{\mathrm{high}}\bigr) \\
&= \Phi\!\left(\frac{y_{\mathrm{high}} - \mu(x)}{\sigma(x)}\right) - \Phi\!\left(\frac{y_{\mathrm{low}} - \mu(x)}{\sigma(x)}\right)
\end{aligned}
$$

- $y_{\mathrm{low}}, y_{\mathrm{high}}$: 目標範囲の下限・上限
- $\Phi$: 標準正規分布の累積分布関数（CDF）
- $\mu(x), \sigma(x)$: GPR の予測平均・予測標準偏差

ノートブック 7 の既定 `settings` 例：発光波長は PTR（600–900 nm）、PLQY は PI（最大化）。

**多目的スコア（ノートブック 7）** — 各目的の達成確率の **log 和** を最大化（独立とみなす）

$$
A_7(x) = \sum_{k=1}^{K} \log P_k(x), \qquad
\text{選抜: } \arg\max_x A_7(x)
$$

- $K$: 目的変数の数（本リポジトリでは 2）
- $P_k(x)$: 目的 $k$ に対する達成確率（PI または PTR）

---

#### ノートブック 8 で追加する成分

**EI（Expected Improvement, 最大化）** — PI と同じ $f^*$ を用いる期待改善量

$$
\delta(x) = \mu(x) - f^*, \quad z(x) = \frac{\delta(x)}{\sigma(x)}, \qquad
\mathrm{EI}(x) = \max\!\left(0,\; \delta(x)\,\Phi(z) + \sigma(x)\,\phi(z)\right)
$$

- $\delta(x)$: 現在の閾値 $f^*$ に対する予測平均の上振れ量
- $z(x)$: 標準化改善量
- $\Phi$: 標準正規分布の累積分布関数（CDF）
- $\phi$: 標準正規分布の確率密度関数（PDF）
- $\mu(x), \sigma(x)$: GPR の予測平均・予測標準偏差

---

**UCB（Upper Confidence Bound, 最大化）**

$$
\mathrm{UCB}(x) = \mu(x) + \kappa\,\sigma(x) \qquad (\kappa = 2.0 \text{ を既定})
$$

- $\kappa$: 探索寄りか活用寄りかを調整する係数（大きいほど探索寄り）
- $\mu(x), \sigma(x)$: GPR の予測平均・予測標準偏差

---

**constrained EI（ノートブック 8）** — 発光波長は PTR、PLQY は EI の **積**による獲得スコア

$$
A_{\mathrm{CEI}}(x) = P_{\mathrm{PTR}}^{\mathrm{em}}(x) \times \mathrm{EI}^{\mathrm{PLQY}}(x)
$$

- $P_{\mathrm{PTR}}^{\mathrm{em}}(x)$: 発光波長が目標範囲に入る確率
- $\mathrm{EI}^{\mathrm{PLQY}}(x)$: PLQY の期待改善量
- $A_{\mathrm{CEI}}(x)$: 制約確率で重み付けした期待改善スコア（同時発生確率そのものではない）

---

**weighted UCB（ノートブック 8）** — 各ラウンドの候補集合 $\mathcal{C}$ 上で UCB を z スコア化し加重和

$$
\mathrm{UCB}_k(x) = \mu_k(x) + \kappa\,\sigma_k(x), \qquad
z_k(x) = \frac{\mathrm{UCB}_k(x) - \mathrm{mean}_{x' \in \mathcal{C}}(\mathrm{UCB}_k)}{\mathrm{std}_{x' \in \mathcal{C}}(\mathrm{UCB}_k) + 10^{-12}}
$$

$$
A_{\mathrm{WUCB}}(x) = w_{\mathrm{PLQY}}\, z_{\mathrm{PLQY}}(x) + w_{\mathrm{em}}\, z_{\mathrm{em}}(x)
\qquad (\text{既定: } w_{\mathrm{PLQY}} = w_{\mathrm{em}} = 0.5)
$$

- $\mathcal{C}$: そのラウンドで評価対象にしている候補集合
- $z_k(x)$: 目的 $k$ の UCB を候補集合内で標準化した値
- $w_{\mathrm{PLQY}}, w_{\mathrm{em}}$: 各目的の重み
- $\kappa$: UCB の探索係数

| 記号 | ノート7 | ノート8 (`constrained_ei`) | ノート8 (`weighted_ucb`) |
|---|---|---|---|
| 発光波長 | $P_{\mathrm{PTR}}$（600–900 nm） | $P_{\mathrm{PTR}}$ | $\mathrm{UCB}$ → $z_{\mathrm{em}}$ |
| PLQY | $P_{\mathrm{PI}}$ | $\mathrm{EI}$ | $\mathrm{UCB}$ → $z_{\mathrm{PLQY}}$ |
| 合成 | $\sum \log P_k$ | 積 | 加重和 |

---

## 実験（ノートブック構成）

ノートブックは番号順に実行することを想定している。6 番以降は BO 系の参考・本番実装、1–5 番はデータ準備から DoE 系候補抽出まで。

| # | ファイル | 内容 |
|---|---|---|
| 1 | `notebook/1.Preparation.ipynb` | Deep4Chem データ読み込み、SMILES から RDKit 記述子を計算。`data/dataset_train.csv`, `dataset_test.csv` を生成 |
| 2 | `notebook/2.EDA.ipynb` | 目的変数の分布、記述子との相関、PCA 等による探索的データ解析 |
| 3 | `notebook/3.Regression.ipynb` | LightGBM と GPR の比較・評価（CV 指標、カーネル選択の参考） |
| 4 | `notebook/4.Finalize_LGB.ipynb` | LightGBM のハイパーパラメータ最適化（Optuna）と最終モデル保存 |
| 5 | `notebook/5.Predict_test_LGB.ipynb` | **DoE 系候補抽出**。LGB 予測 → D 最適サブセット → OCSVM-AD → 閾値フィルタ |
| 6 | `notebook/6.BO_multi-sample.ipynb` | 単一目的 BO の参考実装（発光 **または** PLQY のどちらか一方） |
| 7 | `notebook/7.BO_multi_y_multi_sample.ipynb` | **多目的 BO（PTR/PI）**。GPR + log 確率和で 20 件を逐次選抜 |
| 8 | `notebook/8.BO_multi_y_multi_sample.ipynb` | **多目的 BO（EI/UCB）**。`constrained_ei` または `weighted_ucb` で 20 件を逐次選抜 |

### データ
ここからデータセットをダウンロードすることが可能。

https://figshare.com/articles/dataset/DB_for_chromophore/12045567/2?file=23637518

↑上記URLからcsvをダウンロードして、そのcsvを`data`ディレクトリに入れてから、ノートブック1を実行すること。ノートブック内で学習データとテストデータに分割する。

※元のDatasetには`Solvent`という列があり、溶液の発光波長を測定した際の溶媒の種類が記載されているが、今回はこの列は考慮しなかった。理由としては、材料によっては溶媒により発光波長が変化するものがあるが、おおよそどの程度の波長を示すかの材料自身のポテンシャルが知りたいため。厳密に考慮する場合は、ダミー変数などに変換して説明変数として扱うこと。

- **学習データ**: 発光波長・PLQY の両方が揃っている行（約 13,000 化合物）
- **テスト候補**: PL データが欠損している行（約 7,000 化合物）— ここから次の実験候補を選ぶ
- **記述子**: RDKit記述子 or Mordred記述子（好きなほうを選んでよいが、Mordred記述子は次元が大きいので計算時間がかかる）

### 出力

BO 系ノートブック（7, 8）の結果は `notebook/outputs/result/` 以下に CSV として保存される。

| ファイル例 | 内容 |
|---|---|
| `next_samples_*.csv` | 選抜された候補の記述子 |
| `bo_selected_summary_*.csv` | 各ラウンドの予測平均・標準偏差・獲得スコア |
| `estimated_y_prediction_*.csv` | 全候補に対する GPRによる予測値 |
| `estimated_y_prediction_multi_y_std_*.csv` | 全候補に対する GPRによる標準偏差 |

---

## 結果

以下は各ノートブックを既定設定で実行したときの**たたき台レベル**の比較である。数値はノートブック内の出力・保存 CSV に基づく。

### 比較概要

| 項目 | Notebook 5（DoE / LGB） | Notebook 7（BO / PI-PTR） | Notebook 8（BO / EI-UCB） |
|---|---|---|---|
| 予測モデル | LightGBM | GPR（単一カーネル） | GPR（単一カーネル） |
| 候補選択 | D 最適 → AD → 閾値フィルタ（事後的） | log(P(達成)) の和 — 逐次 BO | constrained EI / weighted UCB — 逐次 BO |
| 入力候補数 | 3,500（D 最適で 7,104 から間引き） | 7,104（全テスト） | 7,104（全テスト） |
| 最終候補数 | **23** | **20** | **20** |
| 不確かさの利用 | なし（AD のみ） | GPR 標準偏差 → PI/PTR | GPR 標準偏差 → EI/UCB |
| 多目的の扱い | 予測後に AND 条件でフィルタ | 確率の log 和 | EI × 制約確率、または UCB 加重和 |

### 閾値条件（共通の目標仕様）

Notebook 5 では次の条件を **予測値** に対して適用している。

- Emission max ≥ **600 nm**
- Quantum yield ≥ **0.5**（50 %）

Notebook 7 では `settings` により、発光波長 **600–900 nm（PTR）**、PLQY **（PI, 最大化）** を設定している。Notebook 8 の `constrained_ei` では PLQY の EI に発光波長 600–900 nm の範囲確率を掛け合わせる。

### 考察
**[Notebook 5_DoE]**     
23種類の候補が抽出されたが、溶媒違いを除くと実質6種類の構造であった。代表構造を以下に示す。     
発光波長と量子収率の両方を考慮すると、`17908` が有力候補になりうる。
![DoE materials](./images/DoE.png)

**[Notebook 7_PTR]**     
発光波長・量子収率の両方で PTR を用い、両者の積が大きい候補を20種類抽出した。実際には、異なる構造は3種類のみだった。代表構造を以下に示す。`14785` は獲得関数の積が最も高かったが、予測波長は短波側だった。`3186` は次点候補として抽出されたが、発光波長は長波側である一方、量子収率は低かった。`289` は DoE 手法でも抽出された候補である。
![PTR materials](./images/PTR_PTR.png)

**[Notebook 7_PI]**      
発光波長には PI（最良値更新確率）を、量子収率には PTR を用い、両者の積が大きい候補を20種類抽出した。この手法では候補の多様性が高く、溶媒違いを除いて同一構造の重複は見られなかった。代表候補を以下に示す。`17407` は短波側予測である一方、獲得関数の積は最も高かった。抽出候補の量子収率は全体として低め（0.35程度）だった。PI の値は PTR より小さく（PTR は約0.5、PI 最大は約0.1）、予測標準偏差も大きいため、探索寄りの選抜になっていると考えられる。
![PI materials](./images/PI_PTR.png)

**[Notebook 8_UCB]**  
この手法でも20種類の多様な候補が選ばれたが、選抜傾向は Notebook 7_PI に近かった。代表候補を以下に示す。最も weighted UCB が高かったのは、PI でも選ばれた `17407` である。ただし、この候補の `std` は 338 nm と非常に大きく、不確実性項によりスコアが押し上げられている。つまり、予測値そのものよりも不確実性が強く効いて選ばれた可能性が高い。一方、`13136` や `1838` は予測波長が長波側で、`std` も 60–70 nm 程度であり、`17407` よりスコアは低いものの比較的妥当な候補と考えられる。なお、予測波長が2番目に長かった `1838` は PI でも選出されている。
![UCB materials](./images/UCB.png)

**[Notebook 8_EI]**  
最後に EI ベースの結果を示す。同一構造の候補がいくつか選ばれているが、Notebook 7_PTR よりは分散していた。`14758` が最も獲得スコア（EI × PTR）が高かった。この手法では 700 nm 以上の予測値は得られず、最長でも `19325` の 674 nm だった。全体としては探索性が強すぎず、比較的活用寄りの結果となった。
![EI materials](./images/EI.png)


## 結論

- **DoE 系** — 発光波長が700nmを越す材料は見らなかったが、量子収率とのバランスがとれた候補は選定できた
- **BO 系** — 発光波長が700nmを越す材料は見られたが、獲得関数の種類によってはかなり探索寄りの結果となるため注意が必要である

## 環境構築

### 前提

- [Conda](https://docs.conda.io/) または [Mamba](https://mamba.readthedocs.io/) が利用できること
- Jupyter Lab / Notebook が起動できること

### Conda 環境の作成

リポジトリルートで `environment.yml` から環境を作成する。

```bash
conda env create -f environment.yml
conda activate chem_win
```

`environment.yml` の主な依存関係:

| カテゴリ | パッケージ |
|---|---|
| Python | 3.11 |
| 数値・ML | numpy, pandas, scipy, scikit-learn, lightgbm, joblib |
| 化学情報 | rdkit, mordred |
| 可視化 | matplotlib, seaborn, py3dmol |
| 最適化 | optuna |

### 実行手順（最小例）

```bash
conda activate chem_win
cd notebook
jupyter lab
```

1. `1.Preparation.ipynb` から順に実行（記述子計算は時間がかかる）
2. DoE 系のみ: `4.Finalize_LGB.ipynb` → `5.Predict_test_LGB.ipynb`
3. BO 系のみ: `6`（任意）→ `7` または `8`

## 引用

本リポジトリのベイズ最適化・実験計画法の考え方は、次の書籍・サンプルコードに基づく。

> 金子弘昌『**Pythonで学ぶ実験計画法入門 — ベイズ最適化によるデータ解析**』（KS情報科学専門書）  
> [Amazon.co.jp リンク](https://www.amazon.co.jp/%EF%BC%B0%EF%BD%99%EF%BD%84%EF%BD%88%EF%BD%8F%EF%BD%8E%E3%81%A7%E5%AD%A6%E3%81%B6%E5%AE%9F%E9%A8%93%E8%A8%88%E7%94%BB%E6%B3%95%E5%85%A5%E9%96%80-%E3%83%99%E3%82%A4%E3%82%BA%E6%9C%80%E9%81%A9%E5%8C%96%E3%81%AB%E3%82%88%E3%82%8B%E3%83%87%E3%83%BC%E3%82%BF%E8%A7%A3%E6%9E%90-%EF%BC%AB%EF%BC%B3%E6%83%85%E5%A0%B1%E7%A7%91%E5%AD%A6%E5%B0%82%E9%96%80%E6%9B%B8-%E9%87%91%E5%AD%90%E5%BC%98%E6%98%8C-ebook/dp/B09C89HZRV)
