#!/usr/bin/env python
import json
import os

import click
import networkx as nx


@click.command(help="Generates relational data in form of graphs. OUT_FILE defines where to save the output.")
@click.argument('out_file', type=str)
@click.option('--num_nodes', '-n', type=int, default=15, help='Number of nodes of the graph.')
@click.option('--edge_probability', '-p', type=float, default=0.0463, help='The probability of a node having an edge.')
@click.option('--random_seed', '-r', type=int, default=42,
              help='Random seed used for reproducibility. Best not to touch.')
def main(out_file, num_nodes, edge_probability, random_seed):
    base_realpath = os.path.realpath(os.path.splitext(out_file)[0])
    kb_file = f"{base_realpath}.pl"
    dataset_file = f"{base_realpath}.json"

    graph = nx.generators.random_graphs.binomial_graph(num_nodes, edge_probability, random_seed, True)

    pos_ex = []
    for node in graph.nodes:
        targets = list(nx.dfs_preorder_nodes(graph, node))
        for target in targets:
            pos_ex.append({"X": f"n{node}", "Y": f"n{target}"})

    neg_ex = []
    for node in graph.nodes:
        for non_target in graph.nodes:
            if not {"X": f"n{node}", "Y": f"n{non_target}"} in pos_ex:
                neg_ex.append({"X": f"n{node}", "Y": f"n{non_target}"})

    with open(dataset_file, "w") as f:
        json.dump({'pos': pos_ex, 'neg': neg_ex}, f)

    kb = []
    for source, target in graph.edges:
        kb.append(f"connected(n{source}, n{target}) .")
    for node in graph.nodes:
        kb.append(f"connected(n{node}, n{node}) .")

    with open(kb_file, 'w') as f:
        f.write("\n".join(kb))

    click.echo(f"Generated {len(pos_ex)} positive examples.")
    click.echo(f"Generated {len(neg_ex)} negative examples.")


if __name__ == '__main__':
    main()
