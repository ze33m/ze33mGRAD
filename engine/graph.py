from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})
    
    for n in nodes:
        label = '''<
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
          <TR>
            <TD COLSPAN="2">{label}</TD>
          </TR>
          <TR>
            <TD>data</TD>
            <TD>grad</TD>
          </TR>
          <TR>
            <TD>{data}</TD>
            <TD>{grad}</TD>
          </TR>
        </TABLE>>'''.format(label=n.label if hasattr(n, 'label') else '', 
                           data=n.data if hasattr(n, 'data') else '', 
                           grad=n.grad if hasattr(n, 'grad') else '')
        
        dot.node(name=str(id(n)), label=label, shape='plaintext')
        
        if hasattr(n, '_op') and n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op, 
                    shape='ellipse', fillcolor='lightgray')
            dot.edge(str(id(n)) + n._op, str(id(n)))
    
    for n1, n2 in edges:
        if hasattr(n2, '_op') and n2._op:
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)
        else:
            dot.edge(str(id(n1)), str(id(n2)))
    
    return dot