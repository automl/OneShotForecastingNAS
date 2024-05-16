"""
 Network architecture visualizer using graphviz
 adapted from https://github.com/khanrc/pt.darts/blob/master/visualize.py
"""
import sys
from graphviz import Digraph
from tsf_oneshot.cells.utils import check_node_is_connected_to_out


def plot(n_nodes: int,
         n_input_nodes: int,
         operations: list[int],
         has_edges: list[bool],
         PRIMITIVES: list[str],
         file_path, caption=None):
    """ make DAG plot and save to file_path as .png """
    edge_attr = {
        'fontsize': '20',
        'fontname': 'times'
    }
    node_attr = {
        'style': 'filled',
        'shape': 'rect',
        'align': 'center',
        'fontsize': '20',
        'height': '0.5',
        'width': '0.5',
        'penwidth': '2',
        'fontname': 'times'
    }
    g = Digraph(
        format='png',
        edge_attr=edge_attr,
        node_attr=node_attr,
        engine='dot')
    g.body.extend(['rankdir=LR'])

    # input nodes
    for i in range(n_input_nodes):
        g.node(f'(in) {i}', fillcolor='darkseagreen2')

    max_nodes = n_nodes + n_input_nodes

    g.node(f"(out) {max_nodes - 1}", fillcolor='palegoldenrod')

    k = 0
    all_edges = {}
    for i in range(n_input_nodes, max_nodes):
        # The first 2 nodes are input nodes
        for j in range(i):
            if has_edges[k]:
                if i == max_nodes - 1:
                    end = f"(out) {max_nodes - 1}"
                else:
                    end = str(i)
                if j < n_input_nodes:
                    start = f'(in) {j}'
                else:
                    start = str(j)
                all_edges[f'{i}<-{j}'] = (start, end, PRIMITIVES[operations[k]])
            k += 1

    nodes_to_remove = set(range(n_input_nodes, max_nodes - 1))
    for i in range(n_input_nodes, max_nodes - 1):
        check_node_is_connected_to_out(i, n_nodes_max=max_nodes, nodes_to_remove=nodes_to_remove,
                                       edges=all_edges)

    for i in range(n_input_nodes, max_nodes - 1):
        if i not in nodes_to_remove:
            g.node(str(i), fillcolor='lightblue')

    edges_to_remove = set()
    for node_to_remove in nodes_to_remove:
        for edge in all_edges.keys():
            edge_nodes = edge.split('<-')
            if str(node_to_remove) in edge_nodes:
                edges_to_remove.add(edge)

    for edge2remove in edges_to_remove:
        all_edges.pop(edge2remove)

    for k, value in all_edges.items():
        g.edge(value[0], value[1], label=value[2], fillcolor="gray")

    # add image caption
    if caption:
        g.attr(label=caption, overlap='false', fontsize='20', fontname='times')

    g.render(file_path, view=True)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError("usage:\n python {} GENOTYPE".format(sys.argv[0]))

    genotype_str = sys.argv[1]
    try:
        genotype = gt.from_str(genotype_str)
    except AttributeError:
        raise ValueError("Cannot parse {}".format(genotype_str))

    plot(genotype.normal, "normal")
    plot(genotype.reduce, "reduction")
