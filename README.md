# U-DyGAD: Unified Dynamic Graph Anomaly Detection

A unified spatio-temporal framework for simultaneous **node-level** and **edge-level** anomaly detection in dynamic graphs. U-DyGAD combines GCN-based spatial encoding with transformer-based temporal encoding, applies MRQSampling for anomaly-aware subgraph extraction, and uses GraphStitch for cross-level information transfer.

**Paper:** *U-DyGAD: Unified Dynamic Graph Anomaly Detection via Spatio-Temporal Representation Learning*
**Authors:** Pritam Das, Fahim Sadik Rashad — Bangladesh University of Engineering & Technology (BUET)

---

## Key Contributions

- **Unified multi-level detection** — jointly detects node and edge anomalies in dynamic graphs
- **MRQSampling in dynamic context** — anomaly-aware subgraph extraction via maximum Rayleigh quotient (first application to dynamic graphs)
- **Spatio-temporal encoding** — GCN spatial encoder + transformer temporal encoder
- **GraphStitch** — cross-level information transfer between node and edge tasks with learnable mixing weights
- **Multi-dataset generalization** — evaluated on Reddit, Bitcoin-OTC, Bitcoin-Alpha

## Architecture

1. **Random temporal sampling** — partition the graph into temporal snapshots simulating dynamic behavior
2. **MRQSampling** — extract subgraphs with maximum Rayleigh quotient preserving anomaly structure
3. **Spatio-temporal encoder** — 2-layer GCN for spatial features, transformer for temporal features across snapshots
4. **GraphStitch** — separate MLPs per task level with learnable cross-level mixing, followed by anomaly scoring

## Installation

```bash
git clone https://github.com/pritamdas99/ext-unified-gad.git
cd ext-unified-gad
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.2+, DGL, PyTorch Geometric 2.5+, scikit-learn, CUDA 12 (for GPU)

## Dataset Setup

Download the benchmark datasets from [Stanford SNAP](https://snap.stanford.edu/data/):

```bash
mkdir -p datasets/csv

# Bitcoin-OTC
wget https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz -O datasets/csv/soc-sign-bitcoinotc.csv.gz
gunzip -f datasets/csv/soc-sign-bitcoinotc.csv.gz

# Bitcoin-Alpha
wget https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz -O datasets/csv/soc-sign-bitcoinalpha.csv.gz
gunzip -f datasets/csv/soc-sign-bitcoinalpha.csv.gz

# Reddit Hyperlinks
wget https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv -O datasets/csv/soc-redditHyperlinks-body.tsv
```

### Dataset Overview

| Index | Dataset | Description | Format |
|-------|---------|-------------|--------|
| 0 | reddit | Social network (posts linked by keywords, Word2Vec features) | DGL |
| 1 | weibo | Social network | DGL |
| 20 | bitcoin-otc | Who-trusts-whom Bitcoin trading network | CSV |
| 21 | bitcoin-alpha | Bitcoin trust rating network | CSV |
| 22 | reddit-hyperlinks | Reddit community hyperlinks | CSV |

DGL-format datasets (reddit, weibo, etc.) go in `datasets/edge_labels/`. Full dataset index is in `src/utils.py`.

## Usage

All commands run from the `src/` directory.

```bash
cd src

# Bitcoin-OTC (index 20), 1-hop, node+edge detection, 5 trials
python main.py --datasets 20 --pretrain_model graphmae --kernels gcn \
    --lr 5e-4 --save_model --epoch_pretrain 50 --batch_size 10 \
    --khop 1 --epoch_ft 300 --lr_ft 0.003 --final_mlp_layers 3 \
    --cross_modes ne2ne --metric AUROC --trials 5

# Reddit (index 0)
python main.py --datasets 0 --pretrain_model graphmae --kernels gcn \
    --lr 5e-4 --save_model --epoch_pretrain 50 --batch_size 10 \
    --khop 1 --epoch_ft 300 --lr_ft 0.003 --final_mlp_layers 3 \
    --cross_modes ne2ne --metric AUROC --trials 5

# Multiple datasets (range)
python main.py --datasets 20-21 --khop 1 --cross_modes ne2ne
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--datasets` | — | Dataset index or range (e.g., `20`, `0-2`) |
| `--khop` | — | Neighborhood hops: `1` (star+norm) or `2` (convtree+norm) |
| `--cross_modes` | `ng2ng` | Task routing, e.g. `ne2ne` (node+edge -> node+edge) |
| `--pretrain_model` | `graphmae` | Pretrain encoder type |
| `--kernels` | `gcn` | GNN backbone type |
| `--lr` | `0.01` | Pretrain learning rate |
| `--lr_ft` | `0.003` | Fine-tuning learning rate |
| `--epoch_pretrain` | `100` | Pretrain epochs |
| `--epoch_ft` | `200` | Fine-tuning epochs |
| `--patience` | `50` | Early stopping patience |
| `--batch_size` | `8` | Batch size |
| `--hid_dim` | `32` | Hidden dimension |
| `--n_heads` | `4` | Transformer attention heads |
| `--n_layers_attention` | `2` | Number of transformer layers |
| `--final_mlp_layers` | `2` | MLP layers in final predictor |
| `--metric` | `AUROC` | Evaluation metric (`AUROC`, `MacroF1`, `AUPRC`) |
| `--trials` | `1` | Number of experiment trials |
| `--save_model` | — | Flag to save trained model |
| `--device` | `cuda` | Device (`cuda` or `cpu`) |

## Results

### Node Anomaly Detection (AUROC)

| Method | Bitcoin-OTC | Reddit |
|--------|-------------|--------|
| Netwalk | 0.9504 | 0.7821 |
| MTHL | 0.9353 | 0.7074 |
| SAD | 0.6341 | 0.8551 |
| CLDG | 0.8394 | 0.8348 |
| **U-DyGAD** | **0.9991** | **0.9840** |

### Edge Anomaly Detection (AUROC)

| Method | Bitcoin-Alpha | Bitcoin-OTC |
|--------|---------------|-------------|
| ADDGRAPH | 0.8369 | 0.8477 |
| TADDY | 0.9423 | 0.9425 |
| StrGNN | 0.8627 | 0.8836 |
| TGAT | 0.8398 | 0.8755 |
| **U-DyGAD** | **0.9322** | **0.9772** |

## Project Structure

```
src/
├── main.py           # Entry point — dataset selection, training loop
├── e2e_model.py      # UnifyMLPDetector — train/eval, multi-task loss with Pareto weighting
├── encoder.py        # GCN spatial encoder
├── transformer.py    # Transformer temporal encoder
├── combine.py        # GCNTemporalFusion — spatial + temporal fusion
├── predictor.py      # UNIMLP_E2E — end-to-end model with MLP heads
├── Pareto_fn.py      # Pareto multi-task loss balancing
└── utils.py          # Dataset class, MRQSampling, argument parsing, data loading
datasets/
├── edge_labels/      # DGL format datasets (reddit, weibo, etc.)
├── unified/          # Graph collection datasets
└── csv/              # CSV edge-list datasets (bitcoin-otc, bitcoin-alpha)
```



## References

- **UniGAD** — Lin et al., *Towards Graph Anomaly Detection via Unified Multi-level Subgraph Sampling and Contrastive Learning*
- **GeneralDyG** — Yang et al., *Generalizable Dynamic Graph Anomaly Detection*
- **ADDGRAPH** — Zheng et al., *AddGraph: Anomaly Detection in Dynamic Graph*
- **TADDY** — Liu et al., *Anomaly Detection in Dynamic Graphs via Transformer*
- **StrGNN** — Cai et al., *Structural Temporal Graph Neural Networks for Anomaly Detection*
