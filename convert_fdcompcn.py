"""
    Convert FDCompCN DGL dataset to the format expected by CARE-GNN's load_data().

    Input:  data/FDCompCN/  (unzipped from FDCompCN.zip — contains .bin DGL graph files)
    Output: data/FDCompCN.mat-style pickles and npz files:
            - data/comp_homo_adjlists.pickle
            - data/comp_rel1_adjlists.pickle  (C-I-C: investment)
            - data/comp_rel2_adjlists.pickle  (C-S-C: supplier)
            - data/comp_rel3_adjlists.pickle  (C-P-C: customer)
            - data/comp_features.npy
            - data/comp_labels.npy

    Usage:
        1. Unzip FDCompCN.zip into data/FDCompCN/
        2. python convert_fdcompcn.py
        3. Now you can run: python run_experiments.py --datasets comp

    Requirements: pip install dgl (in addition to existing deps)
"""

import os
import pickle
import glob
import numpy as np
import scipy.sparse as sp
from collections import defaultdict

try:
    import dgl
    from dgl import load_graphs
except ImportError:
    raise ImportError(
        "DGL is required to convert FDCompCN. Install with:\n"
        "  pip install dgl -f https://data.dgl.ai/wheels/repo.html\n"
        "or see https://www.dgl.ai/pages/start.html"
    )


def sparse_to_adjlist(sp_matrix):
    """Convert a scipy sparse matrix to adjacency list (dict of sets), with self-loops."""
    homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
    adj_lists = defaultdict(set)
    edges = homo_adj.nonzero()
    for idx, node in enumerate(edges[0]):
        adj_lists[node].add(edges[1][idx])
        adj_lists[edges[1][idx]].add(node)
    return adj_lists


def find_dgl_graph(data_dir):
    """
    Locate the DGL graph file inside data_dir.
    SplitGNN typically saves as .bin files via dgl.save_graphs().
    """
    # look for .bin files first (most common DGL save format)
    bin_files = glob.glob(os.path.join(data_dir, "**", "*.bin"), recursive=True)
    if bin_files:
        return bin_files

    # also check for any file — sometimes named without extension
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if not f.startswith('.') and not f.endswith('.zip'):
                all_files.append(os.path.join(root, f))
    return all_files


def load_dgl_graph(data_dir):
    """
    Load the FDCompCN DGL graph. Handles both single-file and multi-file cases.
    Returns a DGL graph (possibly heterogeneous) and its labels.
    """
    candidates = find_dgl_graph(data_dir)

    if not candidates:
        raise FileNotFoundError(
            f"No graph files found in {data_dir}. "
            f"Make sure to unzip FDCompCN.zip into this directory."
        )

    print(f"Found candidate files: {candidates}")

    # try loading each file until one works
    for fpath in candidates:
        try:
            graphs, label_dict = load_graphs(fpath)
            print(f"Successfully loaded: {fpath}")
            print(f"  Number of graphs: {len(graphs)}")
            print(f"  Label dict keys: {list(label_dict.keys()) if label_dict else 'None'}")
            return graphs, label_dict, fpath
        except Exception as e:
            print(f"  Failed to load {fpath}: {e}")
            continue

    raise RuntimeError(f"Could not load any DGL graph from {data_dir}")


def inspect_graph(g):
    """Print detailed info about the DGL graph structure."""
    print("\n=== Graph Inspection ===")

    if g.is_homogeneous:
        print(f"Homogeneous graph: {g.num_nodes()} nodes, {g.num_edges()} edges")
        print(f"  Node data keys: {list(g.ndata.keys())}")
        print(f"  Edge data keys: {list(g.edata.keys())}")
    else:
        print(f"Heterogeneous graph")
        print(f"  Node types: {g.ntypes}")
        print(f"  Edge types (canonical): {g.canonical_etypes}")
        for ntype in g.ntypes:
            print(f"  [{ntype}] {g.num_nodes(ntype)} nodes, ndata keys: {list(g.nodes[ntype].data.keys())}")
        for etype in g.canonical_etypes:
            print(f"  {etype}: {g.num_edges(etype)} edges, edata keys: {list(g.edges[etype].data.keys())}")

    print("========================\n")


def extract_from_heterogeneous(g):
    """
    Extract features, labels, and per-relation adjacency from a heterogeneous DGL graph.
    FDCompCN has: C-I-C (investment), C-S-C (supplier), C-P-C (customer).
    """
    # FDCompCN has a single node type (company)
    # Get the node type — should be just one
    assert len(g.ntypes) == 1, f"Expected 1 node type, got {g.ntypes}"
    ntype = g.ntypes[0]
    num_nodes = g.num_nodes(ntype)

    # extract features
    feat_keys = list(g.nodes[ntype].data.keys())
    print(f"Node feature keys: {feat_keys}")

    features = None
    for key in ['feat', 'feature', 'features', 'h', 'x']:
        if key in g.nodes[ntype].data:
            features = g.nodes[ntype].data[key].numpy()
            print(f"Using features from ndata['{key}'], shape: {features.shape}")
            break
    if features is None and feat_keys:
        # try the first key that looks like features (2D tensor)
        for key in feat_keys:
            tensor = g.nodes[ntype].data[key]
            if tensor.dim() == 2:
                features = tensor.numpy()
                print(f"Using features from ndata['{key}'], shape: {features.shape}")
                break
    if features is None:
        raise ValueError(f"Could not find feature tensor. Available keys: {feat_keys}")

    # extract labels
    labels = None
    for key in ['label', 'labels', 'y']:
        if key in g.nodes[ntype].data:
            labels = g.nodes[ntype].data[key].numpy().flatten()
            print(f"Using labels from ndata['{key}'], shape: {labels.shape}")
            break
    if labels is None:
        raise ValueError(f"Could not find label tensor. Available keys: {feat_keys}")

    # extract per-relation adjacency matrices
    relation_adjs = {}
    for src_type, etype, dst_type in g.canonical_etypes:
        src, dst = g.edges(etype=(src_type, etype, dst_type))
        src, dst = src.numpy(), dst.numpy()
        adj = sp.csr_matrix(
            (np.ones(len(src)), (src, dst)),
            shape=(num_nodes, num_nodes),
        )
        # make symmetric (undirected)
        adj = adj + adj.T
        adj[adj > 1] = 1
        relation_adjs[etype] = adj
        print(f"Relation '{etype}': {len(src)} directed edges → "
              f"{adj.nnz} entries after symmetrization")

    return features, labels, relation_adjs, num_nodes


def extract_from_homogeneous(g, label_dict=None):
    """
    Extract features, labels, and adjacency from a homogeneous DGL graph.
    If the graph has edge types stored in edata, split into per-relation graphs.
    """
    num_nodes = g.num_nodes()
    ndata_keys = list(g.ndata.keys())
    edata_keys = list(g.edata.keys())
    print(f"Node data keys: {ndata_keys}")
    print(f"Edge data keys: {edata_keys}")

    # extract features
    features = None
    for key in ['feat', 'feature', 'features', 'h', 'x']:
        if key in g.ndata:
            features = g.ndata[key].numpy()
            print(f"Using features from ndata['{key}'], shape: {features.shape}")
            break
    if features is None:
        for key in ndata_keys:
            tensor = g.ndata[key]
            if tensor.dim() == 2:
                features = tensor.numpy()
                print(f"Using features from ndata['{key}'], shape: {features.shape}")
                break
    if features is None:
        raise ValueError(f"Could not find feature tensor. Available: {ndata_keys}")

    # extract labels
    labels = None
    for key in ['label', 'labels', 'y']:
        if key in g.ndata:
            labels = g.ndata[key].numpy().flatten()
            break
    # also check label_dict from load_graphs
    if labels is None and label_dict:
        for key in ['labels', 'label', 'y']:
            if key in label_dict:
                labels = label_dict[key].numpy().flatten()
                break
    if labels is None:
        raise ValueError(f"Could not find labels. ndata: {ndata_keys}, label_dict: {label_dict}")
    print(f"Labels shape: {labels.shape}, fraud: {(labels == 1).sum()}, benign: {(labels == 0).sum()}")

    # extract per-relation adjacency
    src_all, dst_all = g.edges()
    src_all, dst_all = src_all.numpy(), dst_all.numpy()

    relation_adjs = {}
    # check if edge types are stored in edata
    etype_key = None
    for key in ['etype', 'edge_type', 'type', 'rel', 'relation']:
        if key in g.edata:
            etype_key = key
            break

    if etype_key is not None:
        edge_types = g.edata[etype_key].numpy().flatten()
        unique_types = np.unique(edge_types)
        print(f"Found {len(unique_types)} edge types in edata['{etype_key}']: {unique_types}")

        # FDCompCN relations mapping (based on README):
        # Typically: 0 → C-I-C, 1 → C-S-C, 2 → C-P-C
        relation_names = {0: 'cic', 1: 'csc', 2: 'cpc'}
        for etype_id in unique_types:
            mask = edge_types == etype_id
            src_r = src_all[mask]
            dst_r = dst_all[mask]
            adj = sp.csr_matrix(
                (np.ones(len(src_r)), (src_r, dst_r)),
                shape=(num_nodes, num_nodes),
            )
            adj = adj + adj.T
            adj[adj > 1] = 1
            name = relation_names.get(etype_id, f'rel{etype_id}')
            relation_adjs[name] = adj
            print(f"Relation '{name}' (type={etype_id}): {mask.sum()} edges → {adj.nnz} entries")
    else:
        # single relation graph — treat the whole thing as homo
        print("No edge types found — treating as single-relation graph")
        adj = sp.csr_matrix(
            (np.ones(len(src_all)), (src_all, dst_all)),
            shape=(num_nodes, num_nodes),
        )
        adj = adj + adj.T
        adj[adj > 1] = 1
        relation_adjs['homo'] = adj

    return features, labels, relation_adjs, num_nodes


def build_homo_adj(relation_adjs, num_nodes):
    """Build homogeneous adjacency by taking the union of all relation edges."""
    homo = sp.lil_matrix((num_nodes, num_nodes))
    for adj in relation_adjs.values():
        homo = homo + adj
    homo[homo > 1] = 1
    return homo.tocsr()


def main():
    data_dir = 'data/FDCompCN/'
    output_dir = 'data'
    prefix = 'comp'

    if not os.path.isdir(data_dir):
        # try unzipping
        zip_path = 'data/FDCompCN.zip'
        if os.path.exists(zip_path):
            import zipfile
            print(f"Unzipping {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall('data/')
        else:
            raise FileNotFoundError(
                f"Neither {data_dir} nor {zip_path} found. "
                f"Download FDCompCN.zip from "
                f"https://github.com/Split-GNN/SplitGNN/blob/master/data/FDCompCN.zip "
                f"and place it in data/"
            )

    # load the DGL graph
    graphs, label_dict, fpath = load_dgl_graph(data_dir)
    g = graphs[0]  # take the first graph
    inspect_graph(g)

    # extract based on graph type
    if g.is_homogeneous:
        features, labels, relation_adjs, num_nodes = extract_from_homogeneous(g, label_dict)
    else:
        features, labels, relation_adjs, num_nodes = extract_from_heterogeneous(g)

    # ensure we have exactly 3 relations for CARE-GNN
    rel_names = sorted(relation_adjs.keys())
    print(f"\nRelations found: {rel_names}")

    if len(rel_names) < 3:
        print(f"WARNING: Only {len(rel_names)} relations found. "
              f"CARE-GNN expects 3. Will duplicate the last relation.")
        while len(rel_names) < 3:
            last = rel_names[-1]
            dup_name = f"{last}_dup{len(rel_names)}"
            relation_adjs[dup_name] = relation_adjs[last].copy()
            rel_names.append(dup_name)

    # build homo graph (union of all relations)
    homo_adj = build_homo_adj(relation_adjs, num_nodes)

    # convert to adjacency lists (same format as CARE-GNN's sparse_to_adjlist)
    print("\nConverting to adjacency lists...")

    homo_adjlist = sparse_to_adjlist(homo_adj)
    rel_adjlists = [sparse_to_adjlist(relation_adjs[name]) for name in rel_names[:3]]

    # save everything
    os.makedirs(output_dir, exist_ok=True)

    # adjacency lists
    with open(os.path.join(output_dir, f'{prefix}_homo_adjlists.pickle'), 'wb') as f:
        pickle.dump(homo_adjlist, f)
    for i, name in enumerate(rel_names[:3], 1):
        path = os.path.join(output_dir, f'{prefix}_rel{i}_adjlists.pickle')
        with open(path, 'wb') as f:
            pickle.dump(rel_adjlists[i - 1], f)
        print(f"  Saved {path} ({name}: {len(rel_adjlists[i-1])} nodes with neighbors)")

    # features and labels
    np.save(os.path.join(output_dir, f'{prefix}_features.npy'), features)
    np.save(os.path.join(output_dir, f'{prefix}_labels.npy'), labels)

    # print summary
    print(f"\n{'='*60}")
    print(f"FDCompCN conversion complete!")
    print(f"  Nodes:    {num_nodes}")
    print(f"  Features: {features.shape[1]}")
    print(f"  Fraud:    {(labels == 1).sum()} ({(labels == 1).mean()*100:.1f}%)")
    print(f"  Benign:   {(labels == 0).sum()} ({(labels == 0).mean()*100:.1f}%)")
    print(f"  Relations: {rel_names[:3]}")
    for i, name in enumerate(rel_names[:3]):
        print(f"    {name}: {relation_adjs[name].nnz} edges")
    print(f"  Homo:     {homo_adj.nnz} edges")
    print(f"\nFiles saved to {output_dir}/")
    print(f"  {prefix}_homo_adjlists.pickle")
    print(f"  {prefix}_rel1_adjlists.pickle")
    print(f"  {prefix}_rel2_adjlists.pickle")
    print(f"  {prefix}_rel3_adjlists.pickle")
    print(f"  {prefix}_features.npy")
    print(f"  {prefix}_labels.npy")
    print(f"\nYou can now run:")
    print(f"  python run_experiments.py --datasets comp")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()