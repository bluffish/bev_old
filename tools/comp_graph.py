

def draw_graph(start, watch=[]):
    from graphviz import Digraph
    node_attr = dict(style='filled',
                    shape='box',
                    align='left',
                    fontsize='12',
                    ranksep='0.1',
                    height='0.2')
    graph = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    assert(hasattr(start, "grad_fn"))
    if start.grad_fn is not None:
        _draw_graph(start.grad_fn, graph, watch=watch)
    size_per_element = 0.15
    min_size = 12
    # Get the approximate number of nodes and edges
    num_rows = len(graph.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    graph.graph_attr.update(size=size_str)
    graph.render(filename='net_graph.jpg')


def _draw_graph(var, graph, watch=[], seen=[], indent="", pobj=None):
    ''' recursive function going through the hierarchical graph printing off
    what we need to see what autograd is doing.'''
    from rich import print
    if hasattr(var, "next_functions"):
        for fun in var.next_functions:
            joy = fun[0]
            if joy is not None:
                if joy not in seen:
                    label = str(type(joy)).replace(
                        "class", "").replace("'", "").replace(" ", "")
                    label_graph = label
                    colour_graph = ""
                    seen.append(joy)
                    if hasattr(joy, 'variable'):
                        happy = joy.variable
                        if happy.is_leaf:
                            label += " \U0001F343"
                            colour_graph = "green"
                            for (name, obj) in watch:
                                if obj is happy:
                                    label += " \U000023E9 " + \
                                        "[b][u][color=#FF00FF]" + name + \
                                        "[/color][/u][/b]"
                                    label_graph += name
                                    colour_graph = "blue"
                                    break
                            vv = [str(obj.shape[x])
                                for x in range(len(obj.shape))]
                            label += " [["
                            label += ', '.join(vv)
                            label += "]]"
                            label += " " + str(happy.var())
                    graph.node(str(joy), label_graph, fillcolor=colour_graph)
                    print(indent + label)
                    _draw_graph(joy, graph, watch, seen, indent + ".", joy)
                    if pobj is not None:
                        graph.edge(str(pobj), str(joy))