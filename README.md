# RetailRocket 下一件未看过商品推荐

## 项目说明

本项目基于 RetailRocket 匿名电商行为日志，构建面向短会话的两阶段商品推荐流程。系统读取同一会话的前三条浏览或加购行为，先通过商品共现、类目热门和全站热门召回候选商品，再使用机器学习与深度学习模型完成候选排序，输出用户可能继续交互的未见商品。

项目重点解决三类问题：如何将分散行为日志转换为可训练的会话样本，如何降低候选商品排序成本，以及如何避免把用户刚刚交互过的商品再次推荐。项目覆盖数据处理、会话构造、特征工程、候选召回、模型对比、参数调优、批量推荐和 Flask 在线接口，全部使用 CPU，适合作为推荐算法和综合机器学习实践项目。

全量运行处理 `2,756,101` 条行为事件，生成 `69,572` 个有效会话样本。验证集比较 `Popularity`、`Hybrid Recall`、`LR`、`LightGBM Ranker`、`FM`、`MLP` 和 `DeepFM` 后，最终选择 LightGBM Ranker，测试集取得 `Recall@10 = 0.2474`、`NDCG@10 = 0.1469`。

## 实际问题与解决思路

| 实际问题 | 项目处理方式 | 输出 |
| --- | --- | --- |
| 原始日志按事件分散，缺少可训练的会话样本 | 按 `visitorid` 和时间重建会话，以相邻行为间隔超过 30 分钟切分 | 会话级观察窗口和目标商品 |
| 直接对全部商品排序成本较高 | 使用商品共现、类目热门和全站热门进行多路候选召回 | 候选商品表、Candidate Recall |
| 已交互商品再次被推荐缺乏业务意义 | 候选生成阶段统一排除观察窗口中的 3 个商品 | 未看过商品推荐结果 |
| 推荐列表不仅要看是否命中，还要关注顺序 | 对比 LR、LightGBM、FM、MLP 和 DeepFM 等排序模型 | Recall@K、MRR@10、NDCG@10 |
| 离线流程和在线使用方式容易不一致 | 训练、批量推荐和 Flask 接口复用同一套候选特征逻辑 | CSV 结果和 JSON 接口 |

## 数据与问题

数据来源：[RetailRocket recommender system dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)

```text
data/archive/
├── events.csv
├── item_properties_part1.csv
├── item_properties_part2.csv
└── category_tree.csv
```

原始数据字段主要包括访客 ID、事件类型、商品 ID、时间戳、商品属性和类目关系。数据不包含商品名称、图片、价格、库存和用户长期画像，因此项目输出以商品 ID、类目 ID、召回来源和排序结果为主。

全量运行统计如下：

| 数据统计 | 数量 |
| --- | ---: |
| 行为事件 | 2,756,101 |
| 访客 | 1,407,580 |
| 商品属性记录扫描 | 20,275,902 |
| 有效会话样本 | 69,572 |
| 训练 / 验证 / 测试样本 | 55,269 / 6,773 / 7,530 |
| 目标商品种类 | 30,872 |
| 目标行为：浏览 / 加购 / 购买 | 67,952 / 1,516 / 104 |

项目目标定义为：前三条观察行为之后，后续出现的第一个未交互商品。该任务用于评估下一件未见商品推荐能力，不等同于购买率、点击率或用户长期流失预测。

## 数据处理与划分

- 先按访客和事件时间排序，再以相邻行为间隔超过 30 分钟作为会话边界。
- 固定前三条非交易行为作为观察窗口；含交易行为的前三条窗口不直接构造样本。
- 从第四条行为开始查找第一个未出现在观察窗口中的商品，作为目标商品；没有有效目标的会话不进入数据集。
- 按 `visitorid` 固定划分训练集、验证集和测试集，比例约为 `80% / 10% / 10%`，同一访客不会跨集合出现。
- 商品热度、类目热度和商品共现关系只从训练观察窗口构造；验证集用于模型选择，测试集只进行一次最终评估。
- 候选生成阶段过滤前三条已交互商品，避免重复推荐。

会话特征包括浏览次数、加购次数、商品去重数、会话持续时间、行为间隔、开始时间和行为类型。候选排序特征共 23 项，覆盖会话行为、商品热度、类目热度、商品共现、召回来源和类目匹配等信息。

## 数据链路

```text
RetailRocket 原始事件和商品属性
  -> 按 visitorid、时间排序并以 30 分钟切分会话
  -> 取前三条非交易行为构造观察窗口
  -> 查找后续第一个未交互商品作为目标
  -> 按 visitorid 划分 train / val / test
  -> 仅使用训练数据构造热度、类目和共现统计
  -> 商品共现、类目热门、全站热门多路召回
  -> 过滤已交互商品并构造候选排序特征
  -> Popularity / Hybrid Recall / LR / LightGBM / FM / MLP / DeepFM 对比
  -> 按验证集 NDCG@10 选择模型并重新训练最终模型
  -> 测试集评估、批量推荐和 Flask 在线推荐
```

## 模型与训练

### 基线对比

- `Popularity`：按照商品全站热度排序，作为简单基线。
- `Hybrid Recall`：融合商品共现、同类目热门和全站热门的召回分数。
- `LR`：使用标准化候选特征的线性排序模型。
- `LightGBM Ranker`：使用 LambdaRank 学习候选商品的组内相对顺序。
- `FM`：使用 PyTorch 学习商品、类目和行为类型等稀疏字段之间的二阶交互。
- `MLP`：将稀疏字段 Embedding 与 23 项稠密特征结合，学习非线性关系。
- `DeepFM`：融合 FM 的显式二阶交互和 MLP 的高阶非线性表达。

深度模型使用 CPU 训练和推理，采用 `BCEWithLogitsLoss` 与正样本权重处理候选样本中的正负不平衡。所有模型使用相同的候选集和排序指标比较，最终模型由验证集结果决定。

验证集基线结果如下：

| 模型 | Candidate Recall | Recall@5 | Recall@10 | MRR@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Popularity | 0.3228 | 0.0030 | 0.0089 | 0.0022 | 0.0037 |
| Hybrid Recall | 0.3228 | 0.1587 | 0.2266 | 0.1013 | 0.1307 |
| LR | 0.3228 | 0.1688 | 0.2395 | 0.1045 | 0.1361 |
| LightGBM Ranker | 0.3228 | 0.1753 | 0.2461 | 0.1137 | 0.1448 |
| FM | 0.3228 | 0.0627 | 0.1194 | 0.0351 | 0.0544 |
| MLP | 0.3228 | 0.1764 | 0.2458 | 0.1090 | 0.1411 |
| DeepFM | 0.3228 | 0.1211 | 0.1968 | 0.0714 | 0.1005 |

### LightGBM 参数调优

以验证集 `NDCG@10` 为主要选择依据，比较学习率、叶子数、树数量、子节点样本数和采样参数：

| 配置 | learning_rate | num_leaves | n_estimators | min_child_samples | subsample | Val NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 0.03 | 15 | 300 | 60 | 0.85 | **0.1458** |
| B | 0.05 | 31 | 250 | 40 | 0.85 | 0.1448 |
| C | 0.08 | 63 | 180 | 60 | 0.80 | 0.1449 |

确定参数后，使用训练集和验证集重新构造召回统计并训练最终模型，测试集只用于最后一次评估。最终选出的模型为 `LightGBM Ranker`。

## 运行结果

| 指标 | 结果 |
| --- | ---: |
| Candidate Recall | 0.3242 |
| Recall@5 | 0.1809 |
| Recall@10 | **0.2474** |
| MRR@10 | 0.1160 |
| NDCG@10 | **0.1469** |
| Coverage@10 | 0.3916 |

指标含义：`Candidate Recall` 衡量目标商品是否进入候选集；`Recall@K` 衡量前 K 个推荐是否命中目标；`MRR@10` 和 `NDCG@10` 更关注目标商品在推荐列表中的位置；`Coverage@10` 衡量推荐列表覆盖的商品范围。

## 复现

建议使用 Python 3.10 或更高版本，并在 IDE 中选择已经安装项目依赖的解释器。

### 1. 安装依赖

```bash
python -m pip install -r requirements.txt
```

### 2. 准备数据

将数据文件放入仓库相对目录：

```text
data/archive/
├── events.csv
├── item_properties_part1.csv
├── item_properties_part2.csv
└── category_tree.csv
```

程序会自动查找项目目录或项目上级目录中的 `data/archive/`。

### 3. IDE 一键运行

直接运行：

```bash
python run_project.py
```

只需在 `run_project.py` 顶部调整 4 个主要参数：

| 参数 | 默认值 | 可填范围 | 作用 |
| --- | --- | --- | --- |
| `REUSE_EXISTING_MODEL` | `True` | `True` / `False` | 是否复用当前版本已训练模型；关闭后会重建产物并重新训练。 |
| `QUICK_MODE` | `False` | `True` / `False` | 快速检查流程时使用部分数据和较少调优配置；正式结果使用 `False`。 |
| `TOP_K` | `10` | `1~20` | 每个会话输出的推荐商品数量。 |
| `MAX_CANDIDATES` | `50` | `10~200`，且不小于 `TOP_K` | 每个会话保留的候选上限，越大可能提高召回，但会增加训练和预测耗时。 |

### 4. 在线接口

完整训练完成后运行：

```bash
python score_api.py
```

浏览器访问 `http://127.0.0.1:5000/`，也可以调用接口：

```json
{
  "events": [
    {"event": "view", "itemid": 101, "timestamp": "2015-06-01T10:00:00Z"},
    {"event": "view", "itemid": 202, "timestamp": "2015-06-01T10:02:00Z"},
    {"event": "addtocart", "itemid": 303, "timestamp": "2015-06-01T10:05:00Z"}
  ],
  "top_k": 10
}
```

| 地址 | 用途 |
| --- | --- |
| `GET /` | 本地推荐页面 |
| `GET /health` | 查看服务、当前模型和真实示例数量 |
| `GET /examples` | 获取测试集中的真实观察窗口 |
| `POST /recommend` | 接收 3 条 `view` 或 `addtocart` 行为并返回未见商品的 Top-K 推荐 |

### 5. 测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖会话边界、目标商品构造、交易行为隔离、访客级数据划分、候选商品过滤、排序指标、PyTorch 模型预测、模型复用和接口输入校验。

## 项目结构

```text
retailrocket_recommender_v2/
├── run_project.py              # IDE 一键入口和主要参数
├── retailrocket_main.py        # 完整流程编排
├── prepare_data.py             # 会话构造和目标商品生成
├── candidate_generation.py     # 多路召回和候选特征
├── train_models.py             # 召回基线和 ML/DL 模型对比
├── deep_models.py              # CPU PyTorch 模型和稀疏特征编码
├── tune_model.py               # LightGBM 调优和最终模型训练
├── score_batch.py              # 批量 Top-K 推荐
├── score_api.py                # Flask 在线推荐接口
├── templates/recommender.html  # 本地推荐页面
├── tests/                      # 单元测试和接口测试
├── docs/                       # 补充设计说明
├── artifacts/                  # 自动生成的中间产物
├── outputs/                    # 自动生成的模型和结果
└── requirements.txt            # Python 依赖
```

## 输出文件

```text
artifacts/manifest.json
artifacts/recommendation_sessions.parquet
artifacts/candidate_train.parquet
artifacts/candidate_val.parquet
artifacts/candidate_test.parquet
outputs/baseline/baseline_results.csv
outputs/baseline/fm.joblib
outputs/baseline/mlp.joblib
outputs/baseline/deepfm.joblib
outputs/tuning/tuning_results.csv
outputs/model_registry.json
outputs/test_metrics.json
outputs/batch_recommendations.csv
```

## 局限

- 数据集是匿名行为日志，不包含商品名称、图片、价格、库存和用户长期画像，因此推荐结果主要展示商品 ID、类目 ID和召回来源。
- 当前任务是短会话下一件未见商品推荐，不是完整的实时推荐平台，也不等同于购买率或点击率预测。
- 召回阶段决定目标商品能否进入候选集，排序模型无法弥补召回遗漏。
- 商品类目主要根据属性文件和类目树构造，未进一步建模商品价格、库存和时间变化。
- Flask 使用本地开发服务器，仅用于接口演示和推理验证，不代表生产部署方案。

## English Summary

RetailRocket Recommender is a CPU-only two-stage product recommendation project for anonymous e-commerce sessions. It builds leakage-aware session samples, defines the target as the first unseen item after three observed events, recalls candidates from item co-occurrence, category popularity and global popularity, and compares LR, LightGBM Ranker, FM, MLP and DeepFM. LightGBM is selected by validation NDCG@10 and reaches Recall@10 `0.2474` and NDCG@10 `0.1469` on the independent test split, with batch recommendation output and a lightweight Flask API.

## 参考资料

- [RetailRocket recommender system dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
- [LightGBM documentation](https://lightgbm.readthedocs.io/)
- [PyTorch documentation](https://pytorch.org/docs/)
- [Flask documentation](https://flask.palletsprojects.com/)
