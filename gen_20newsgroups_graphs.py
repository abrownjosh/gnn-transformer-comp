import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_undirected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate graph data from 20 Newsgroups for GNN/graph-transformer classification."
        )
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/20newsgroups_graph"),
        help="Directory for embeddings, graph files, and metadata.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="tfidf_svd",
        choices=["tfidf_svd", "sentence_transformer"],
        help="Embedding backend.",
    )
    parser.add_argument(
        "--sentence_model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name (used only with --embedding_model sentence_transformer).",
    )
    parser.add_argument(
        "--svd_dim",
        type=int,
        default=384,
        help="Embedding dimension for tfidf_svd.",
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=120000,
        help="Max vocabulary size for TF-IDF.",
    )
    parser.add_argument(
        "--knn",
        type=int,
        default=25,
        help="Number of nearest neighbors per node (excluding self).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        choices=["cosine", "euclidean"],
        help="Distance metric for kNN graph.",
    )
    parser.add_argument(
        "--edge_kernel",
        type=str,
        default="gaussian",
        choices=["gaussian", "inverse", "binary"],
        help="How to convert kNN distances to edge weights.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.0,
        help=(
            "Sigma for Gaussian kernel. If <= 0, sigma is estimated from the median kNN distance."
        ),
    )
    parser.add_argument(
        "--remove",
        type=str,
        default="headers,footers,quotes",
        help="Comma-separated 20 Newsgroups fields to remove.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_20newsgroups(remove_fields: Tuple[str, ...]):
    train = fetch_20newsgroups(subset="train", remove=remove_fields)
    test = fetch_20newsgroups(subset="test", remove=remove_fields)
    return train, test


def build_tfidf_svd_embeddings(
    train_texts,
    test_texts,
    svd_dim: int,
    max_features: int,
    seed: int,
):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        max_features=max_features,
        min_df=2,
    )

    print("Fitting TF-IDF...")
    x_train_tfidf = vectorizer.fit_transform(train_texts)
    x_test_tfidf = vectorizer.transform(test_texts)

    max_rank = min(x_train_tfidf.shape[0] - 1, x_train_tfidf.shape[1] - 1)
    actual_dim = min(svd_dim, max_rank)
    if actual_dim < 2:
        raise ValueError("SVD dimension is too small after rank constraints.")

    print(f"Running TruncatedSVD to {actual_dim} dimensions...")
    svd = TruncatedSVD(n_components=actual_dim, random_state=seed)
    x_train = svd.fit_transform(x_train_tfidf)
    x_test = svd.transform(x_test_tfidf)

    x_train = normalize(x_train)
    x_test = normalize(x_test)
    return x_train.astype(np.float32), x_test.astype(np.float32)


def build_sentence_transformer_embeddings(train_texts, test_texts, model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for --embedding_model sentence_transformer. "
            "Install with: pip install sentence-transformers"
        ) from exc

    print(f"Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)

    print("Encoding train texts...")
    x_train = model.encode(
        train_texts,
        batch_size=128,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    print("Encoding test texts...")
    x_test = model.encode(
        test_texts,
        batch_size=128,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    return x_train.astype(np.float32), x_test.astype(np.float32)


def distances_to_weights(
    distances: np.ndarray,
    edge_kernel: str,
    sigma: float,
) -> np.ndarray:
    distances = np.asarray(distances, dtype=np.float32)

    if edge_kernel == "binary":
        return np.ones_like(distances, dtype=np.float32)

    if edge_kernel == "inverse":
        return 1.0 / (1.0 + distances)

    if sigma <= 0:
        sigma = float(np.median(distances[distances > 0])) if np.any(distances > 0) else 1.0
    return np.exp(-(distances ** 2) / (2.0 * sigma ** 2)).astype(np.float32)


def build_knn_graph(
    x: np.ndarray,
    k: int,
    metric: str,
    edge_kernel: str,
    sigma: float,
):
    n = x.shape[0]
    if k >= n:
        raise ValueError(f"k must be less than number of nodes ({n}), got k={k}.")

    print(f"Building kNN index: n={n}, k={k}, metric={metric}")
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm="auto")
    nbrs.fit(x)
    distances, indices = nbrs.kneighbors(x)

    indices = indices[:, 1:]
    distances = distances[:, 1:]

    src = np.repeat(np.arange(n, dtype=np.int64), k)
    dst = indices.reshape(-1)
    edge_dist = distances.reshape(-1)
    edge_weight = distances_to_weights(edge_dist, edge_kernel=edge_kernel, sigma=sigma)

    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float32)

    edge_index, edge_weight = to_undirected(edge_index, edge_attr=edge_weight, reduce="mean")
    return edge_index, edge_weight


def save_induced_subgraph(full_data: Data, mask: torch.Tensor, out_path: Path) -> None:
    node_idx = torch.where(mask)[0]
    sub_edge_index, sub_edge_weight = subgraph(
        subset=node_idx,
        edge_index=full_data.edge_index,
        edge_attr=full_data.edge_weight,
        relabel_nodes=True,
    )

    sub_x = full_data.x[node_idx]
    sub_y = full_data.y[node_idx]

    sub_data = Data(x=sub_x, edge_index=sub_edge_index, edge_weight=sub_edge_weight, y=sub_y)
    torch.save(sub_data, out_path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    remove_fields = tuple([s.strip() for s in args.remove.split(",") if s.strip()])
    print("Loading 20 Newsgroups dataset...")
    train_ds, test_ds = load_20newsgroups(remove_fields=remove_fields)

    train_texts = train_ds.data
    test_texts = test_ds.data

    if args.embedding_model == "tfidf_svd":
        x_train, x_test = build_tfidf_svd_embeddings(
            train_texts,
            test_texts,
            svd_dim=args.svd_dim,
            max_features=args.max_features,
            seed=args.seed,
        )
    else:
        x_train, x_test = build_sentence_transformer_embeddings(
            train_texts,
            test_texts,
            model_name=args.sentence_model,
        )

    y_train = np.asarray(train_ds.target, dtype=np.int64)
    y_test = np.asarray(test_ds.target, dtype=np.int64)

    x_all = np.vstack([x_train, x_test]).astype(np.float32)
    y_all = np.concatenate([y_train, y_test], axis=0)

    train_count = x_train.shape[0]
    total_count = x_all.shape[0]

    train_mask = torch.zeros(total_count, dtype=torch.bool)
    train_mask[:train_count] = True
    test_mask = ~train_mask

    edge_index, edge_weight = build_knn_graph(
        x=x_all,
        k=args.knn,
        metric=args.metric,
        edge_kernel=args.edge_kernel,
        sigma=args.sigma,
    )

    data = Data(
        x=torch.from_numpy(x_all),
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=torch.from_numpy(y_all),
        train_mask=train_mask,
        test_mask=test_mask,
    )

    train_emb_path = args.output_dir / "twenty_newsgroups_train_embeddings.npy"
    test_emb_path = args.output_dir / "twenty_newsgroups_test_embeddings.npy"
    all_emb_path = args.output_dir / "twenty_newsgroups_all_embeddings.npy"
    graph_path = args.output_dir / "twenty_newsgroups_full_graph.pt"
    train_graph_path = args.output_dir / "twenty_newsgroups_train_induced_graph.pt"
    test_graph_path = args.output_dir / "twenty_newsgroups_test_induced_graph.pt"
    metadata_path = args.output_dir / "metadata.json"

    np.save(train_emb_path, x_train)
    np.save(test_emb_path, x_test)
    np.save(all_emb_path, x_all)
    torch.save(data, graph_path)

    save_induced_subgraph(data, train_mask, train_graph_path)
    save_induced_subgraph(data, test_mask, test_graph_path)

    metadata = {
        "dataset": "20newsgroups",
        "num_classes": len(train_ds.target_names),
        "class_names": train_ds.target_names,
        "train_nodes": int(train_count),
        "test_nodes": int(total_count - train_count),
        "total_nodes": int(total_count),
        "num_edges_undirected": int(data.edge_index.shape[1] // 2),
        "embedding_dim": int(x_all.shape[1]),
        "embedding_model": args.embedding_model,
        "sentence_model": args.sentence_model if args.embedding_model == "sentence_transformer" else None,
        "knn": int(args.knn),
        "metric": args.metric,
        "edge_kernel": args.edge_kernel,
        "sigma": float(args.sigma),
        "remove_fields": list(remove_fields),
        "seed": int(args.seed),
        "files": {
            "train_embeddings": str(train_emb_path),
            "test_embeddings": str(test_emb_path),
            "all_embeddings": str(all_emb_path),
            "full_graph": str(graph_path),
            "train_induced_graph": str(train_graph_path),
            "test_induced_graph": str(test_graph_path),
        },
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Done.")
    print(f"Saved graph and embeddings under: {args.output_dir}")


if __name__ == "__main__":
    main()
