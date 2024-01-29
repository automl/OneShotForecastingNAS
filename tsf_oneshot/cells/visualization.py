""" Network architecture visualizer using graphviz """
# https://github.com/khanrc/pt.darts/blob/master/visualize.py
import sys
from graphviz import Digraph


def plot( n_nodes: int,
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
        g.node(f'(in) {i}',  fillcolor='darkseagreen2')

    max_nodes = n_nodes + n_input_nodes

    for i in range(n_input_nodes, max_nodes - 1):
        g.node(str(i), fillcolor='lightblue')
    g.node(f"(out) {max_nodes - 1}", fillcolor='palegoldenrod')

    k = 0
    for i in range(n_input_nodes, max_nodes):
        # The first 2 nodes are input nodes
        for j in range(i):
            if has_edges[k]:
                if i == max_nodes - 1:
                    end = f"(out) {max_nodes - 1}"
                else:
                    end = str(i)
                if j < n_input_nodes:
                    start = f'(in) {i}'
                else:
                    start = str(j)
                g.edge(start, end, label=PRIMITIVES[operations[k]], fillcolor="gray")

    # add image caption
    if caption:
        g.attr(label=caption, overlap='false', fontsize='20', fontname='times')

    g.render(file_path, view=False)


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