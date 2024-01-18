import decimal
import string
import traceback

from collections import Counter, deque
from copy import copy, deepcopy
from itertools import chain, combinations, combinations_with_replacement, product
from math import comb, floor
from sys import getsizeof
from typing import Any, Iterable, List, Optional, Set, Tuple, Union

from tqdm import tqdm
from sage.all import Graph, QQ, ZZ


digs = string.digits + string.ascii_letters

ctx = decimal.Context()
ctx.prec = 64


def int2base(x, base):
    """
    Converts a non-negative integer to a specified base.

    Parameters:
    x (int): The non-negative integer to convert.
    base (int): The base for conversion.

    Returns:
    str: The string representation of the integer in the specified base.

    Raises:
    AssertionError: If x is negative.
    """
    
    assert x >= 0
    if x == 0:
        return digs[0]

    digits = []
    while x:
        digits.append(digs[x % base])
        x = x // base

    digits.reverse()

    return ''.join(digits)


def argsort(seq, key=None):
    """
    Returns the indices that would sort an array.

    Parameters:
    seq (iterable): An iterable to be sorted.
    key (function, optional): A function of one argument that is used to extract 
                              a comparison key from each element in seq.

    Returns:
    list: Indices that would sort the array.
    """
    
    return sorted(range(len(seq)), key=lambda i: (key(seq[i]) if key else seq[i]))


def get_pbar(total: Optional[int] = None,
             ncols: int = 80,
             desc: Optional[str] = None,
             leave: bool = True):
    """
    Creates and returns a tqdm progress bar object.

    Parameters:
    total (int, optional): The total number of iterations. If None, total is automatically adjusted.
    ncols (int): The width of the entire output message.
    desc (str, optional): A description to be displayed to the left of the progress bar.
    leave (bool): Whether to leave the progress bar when complete.

    Returns:
    tqdm.std.tqdm: A tqdm progress bar object.
    """

    bf = '{l_bar}{bar}| {n_fmt}/{total_fmt} ({elapsed}<{remaining})'
    return tqdm(total=total,
                ncols=ncols,
                desc=desc,
                leave=leave,
                bar_format=bf)


def pbar(it: Iterable,
         total: Optional[int] = None,
         ncols: int = 80,
         desc: Optional[str] = None,
         leave: bool = True,
         verbose: bool = True):
    """
    Iterates over an Iterable with an optional progress bar.

    Parameters:
    it (Iterable): The iterable to loop over.
    total (int, optional): The total number of iterations. If None, total is automatically adjusted.
    ncols (int): The width of the entire output message.
    desc (str, optional): A description to be displayed to the left of the progress bar.
    leave (bool): Whether to leave the progress bar when complete.
    verbose (bool): Whether to display the progress bar.

    Returns:
    Iterator: An iterator or tqdm iterator over the iterable.
    """

    if verbose:
        bf = '{l_bar}{bar}| {n_fmt}/{total_fmt} ({elapsed}<{remaining})'
        return tqdm(it,
                    total=total,
                    ncols=ncols,
                    desc=desc,
                    leave=leave,
                    bar_format=bf)

    return it


def apply_pool(
        func,
        arguments: Iterable,
        mp: Optional[tuple] = None,
        verbose: bool = True,
        desc: Optional[str] = None):
    """
    Applies a function to all the items in an iterable using multiprocessing.

    Parameters:
    func (function): The function to apply.
    arguments (Iterable): An iterable of arguments to apply the function to.
    mp (tuple, optional): A tuple containing the number of processes and a multiprocessing Pool object.
    verbose (bool): Whether to display a progress bar.
    desc (str, optional): A description for the progress bar.

    Returns:
    list: A list of results from applying the function to each item.
    """
    
    arguments = list(arguments)
    if not isinstance(arguments[0], tuple):
        arguments = [(arg,) for arg in arguments]
    arguments = pbar(arguments, leave=True, desc=desc, verbose=verbose)
    if mp is None:
        return [func(*arg) for arg in arguments]
    _, pool = mp
    return pool.starmap(func, arguments)


def falling_factorial(i: int, j: int):
    """
    Computes the falling factorial of i and j.

    Parameters:
    i (int): The upper value in the falling factorial.
    j (int): The lower value in the falling factorial.

    Returns:
    int: The result of the falling factorial.

    Raises:
    AssertionError: If j is not in the range [0, i].
    """
    
    assert 0 <= j <= i
    if j == 0:
        return 1
    return i * falling_factorial(i-1, j-1)


def multinomial(it: List[int]) -> int:
    """
    Calculate the multinomial coefficient for a given list of integers.

    Args:
        it (List[int]): List of integers giving the sizes of the partitions.

    Returns:
        int: The multinomial coefficient for the given list of integers.
    """
    
    n = sum(it)
    temp = 1
    for k in it:
        temp *= comb(n, k)
        n -= k
    return temp


def get_max_flag_order(nvertices: int, ftype_order: int) -> int:
    """
    Calculates the maximum flag order for a given type order.

    Parameters:
    nvertices (int): The number of vertices in the colorings.
    ftype_order (int): The order of the flag type.

    Returns:
    int: The maximum flag order.
    """
    return int(floor((nvertices) + ftype_order) / 2)


def get_edges(vertices: List[int]) -> List[Tuple[int]]:
    """
    Generates all possible edges for a list of vertices.

    Parameters:
    vertices (List[int]): A list of vertices.

    Returns:
    List[Tuple[int]]: A list of tuples representing edges between vertices.
    """
    edges = [combinations(vertices, 2)]
    edges = [tuple(sorted(edge)) for part in edges for edge in part]

    return edges


def get_nedges(nvertices: int) -> int:
    """
    Calculates the number of edges in a complete graph with a given number of vertices.

    Parameters:
    nvertices (int): The number of vertices in the graph.

    Returns:
    int: The number of edges in the graph.
    """
    return len(get_edges(list(range(nvertices))))


class Coloring:
    """
    A class representing a coloring of the edges of a complete graph.

    Attributes:
    nvertices (int): Number of vertices in the graph.
    ncolors (int): Number of colors used for coloring.
    edge_colors (List[int]): List of colors for each edge.
    ftype_size (int): The size of the flag type.

    Methods:
    from_string(input_string): Creates a Coloring instance from a string representation.
    _clear(): Clears certain internal attributes.
    nvertices: Returns the number of vertices.
    vertices: Returns a list of vertices.
    nedges: Returns the number of edges.
    edges: Returns a list of edges.
    edge_colors: Gets or sets the colors of the edges.
    ncolors: Returns the number of colors.
    colors: Returns a list of colors.
    ftype_size: Gets or sets the flag type size.
    ftype: Returns the flag type.
    used_colors: Returns a set of used colors.
    __hash__(): Returns the hash of the coloring.
    __eq__(other): Checks equality with another Coloring instance.
    __ne__(other): Checks inequality with another Coloring instance.
    __lt__(other): Less than comparison with another Coloring instance.
    __le__(other): Less than or equal to comparison with another Coloring instance.
    __gt__(other): Greater than comparison with another Coloring instance.
    __ge__(other): Greater than or equal to comparison with another Coloring instance.
    __copy__(): Returns a shallow copy of the instance.
    __deepcopy__(memo): Returns a deep copy of the instance.
    __getstate__(): Gets the state of the instance for pickling.
    __setstate__(state): Sets the state of the instance for unpickling.
    __repr__(): Returns the string representation of the instance.
    __str__(): Returns the string representation of the instance.
    __len__(): Returns the number of edges.
    get_edge_color(edge): Returns the color of a given edge.
    set_edge_color(edge, color): Sets the color of a given edge.
    apply_color_map(color_map): Applies a color map to the coloring.
    apply_inverse_color_map(inverse_color_map): Applies an inverse color map to the coloring.
    apply_vertex_map(vertex_map): Applies a vertex map to the coloring.
    apply_inverse_vertex_map(inverse_vertex_map): Applies an inverse vertex map to the coloring.
    subcoloring(indices, as_vertices): Returns a subcoloring based on given indices.
    contains_subcoloring(subcoloring, color_invariant, fixed_vertices, is_canonical): Checks if the coloring contains a given subcoloring.
    _graph_representation(color_invariant): Returns the graph representation of the coloring.
    canonical_representative(algorithm, color_invariant, certificate): Returns the canonical representative of the coloring.
    canonize(algorithm, color_invariant, certificate): Canonizes the coloring.
    automorphisms(algorithm, color_invariant): Returns the automorphisms of the coloring.
    make_ftype(color_invariant): Makes a flag type from the coloring.
    set_ftype(vertices, color_invariant): Sets the flag type for the coloring.
    apply_ftype_homomorphism(hom, color_invariant): Applies a flag type homomorphism to the coloring.
    apply_homomorphism(hom, color_invariant): Applies a homomorphism to the coloring.
    """

    def __init__(
            self,
            nvertices: int,
            ncolors: int = 2,
            edge_colors: Optional[List[int]] = None,
            ftype_size: int = 0):

        self._ncolors = ncolors
        self._nvertices = nvertices
        self._ftype_size = ftype_size
        self._edges = get_edges(self.vertices)

        if edge_colors is not None:
            assert len(edge_colors) == len(self.edges)
            assert all([0 <= color < self.ncolors for color in edge_colors])
            self._edge_colors = edge_colors

        else:
            self._edge_colors = [0 for _ in self.edges]

        self._clear()

    @classmethod
    def from_string(cls, input_string: str):

        input_string_split = input_string.split('.')

        assert input_string_split[0] == 'c'

        ncolors = int(input_string_split[2])
        nvertices = int(input_string_split[3])
        ftype_size = int(input_string_split[4])

        adjacency_info = input_string_split[5]

        if get_nedges(nvertices) == 0:
            assert adjacency_info == '-'
            edge_colors = None
        else:
            adjacency_info = int(adjacency_info, base=36)
            adjacency_info = int2base(adjacency_info, base=ncolors)
            edge_colors = [digs.index(x) for x in str(adjacency_info)]
            n_missing = get_nedges(nvertices)
            n_missing -= len(edge_colors)
            edge_colors = [0 for _ in range(n_missing)] + edge_colors

        coloring = cls(
            nvertices,
            ncolors=ncolors,
            edge_colors=edge_colors,
            ftype_size=ftype_size)

        coloring._str = input_string

        return coloring

    def _clear(self):
        for attr in ['_str', '_gr', '_pt', '_cgr', '_cpt',
                     '_autom', '_cautom']:
            if hasattr(self, attr):
                delattr(self, attr)

    @property
    def nvertices(self) -> int:
        return self._nvertices

    @property
    def vertices(self) -> List[int]:
        return list(range(self._nvertices))

    @property
    def nedges(self) -> int:
        return len(self._edges)

    @property
    def edges(self) -> List[Tuple[int]]:
        return self._edges

    @property
    def edge_colors(self) -> List[int]:
        return self._edge_colors

    @edge_colors.setter
    def edge_colors(self, value: List[int]):
        assert len(value) == len(self)
        assert all([0 <= c < self.ncolors for c in value])
        self._edge_colors = value
        self._clear()

    @property
    def ncolors(self) -> int:
        return self._ncolors

    @property
    def colors(self) -> List[int]:
        return list(range(self._ncolors))

    @property
    def ftype_size(self) -> int:
        return self._ftype_size

    @ftype_size.setter
    def ftype_size(self, value: int):
        self._ftype_size = value
        self._clear()

    @property
    def ftype(self):
        return self.subcoloring([])

    @property
    def used_colors(self):
        return set(self.edge_colors)

    def __hash__(self) -> int:
        return hash(self.__str__())

    def __eq__(self, other) -> bool:
        assert isinstance(other, self.__class__)
        return  self.ncolors == other.ncolors \
            and self.nvertices == other.nvertices \
            and self.edge_colors == other.edge_colors

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __lt__(self, other) -> bool:
        assert isinstance(other, self.__class__)

        for attr in ['ncolors', 'nvertices', 'ftype_size', 'edge_colors']:
            if getattr(self, attr) < getattr(other, attr):
                return True
            elif getattr(self, attr) > getattr(other, attr):
                return False

        return False

    def __le__(self, other) -> bool:
        assert isinstance(other, self.__class__)
        return self < other or self == other

    def __gt__(self, other) -> bool:
        return not self <= other

    def __ge__(self, other) -> bool:
        return not self < other

    def __copy__(self):
        return self.__class__(
            copy(self.nvertices),
            ncolors = copy(self.ncolors),
            edge_colors=copy(self.edge_colors),
            ftype_size=copy(self.ftype_size))

    def __deepcopy__(self, memo):
        return self.__class__(
            deepcopy(self.nvertices),
            ncolors = deepcopy(self.ncolors),
            edge_colors=deepcopy(self.edge_colors),
            ftype_size=deepcopy(self.ftype_size))

    def __getstate__(self):
        return (self.nvertices, self.ncolors, self.edge_colors, self.ftype_size)

    def __setstate__(self, state):
        nvertices, ncolors, edge_colors, ftype_size = state
        self.__init__(
            nvertices,
            ncolors=ncolors,
            edge_colors=edge_colors,
            ftype_size=ftype_size)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        if hasattr(self, '_str'):
            return self._str

        if len(self.edges) > 0:
            adjacency_info = ''.join([digs[x] for x in self.edge_colors])
            adjacency_info = int(adjacency_info, base=self.ncolors)
            adjacency_info = int2base(adjacency_info, base=36)

        else:
            adjacency_info = '-'

        output = "c"

        output += f".2" # the uniformity
        output += f".{self.ncolors}"
        output += f".{self.nvertices}"
        output += f'.{self.ftype_size}'
        output += f".{adjacency_info}"

        self._str = output

        return output

    def __len__(self) -> int:
        return len(self.edges)

    def get_edge_color(self, edge: Tuple[int]) -> int:
        edge = tuple(sorted(edge))
        return self._edge_colors[self._edges.index(edge)]

    def set_edge_color(self, edge: Tuple[int], color: int) -> None:
        idx = self.edges.index(edge)
        if color != self.edge_colors[idx]:
            self.edge_colors[idx] = color
            self._clear()

    def apply_color_map(self, color_map: List[int]) -> None:
        assert sorted(color_map) == list(self.colors)
        self.edge_colors = [color_map[c] for c in self.edge_colors]
        self._clear()

    def apply_inverse_color_map(self, inverse_color_map: List[int]):
        color_map = argsort(inverse_color_map)
        return self.apply_color_map(color_map)

    def apply_vertex_map(self, vertex_map: List[int]) -> None:
        assert sorted(vertex_map) == self.vertices
        new_edge_order = get_edges(vertex_map)
        new_edge_order = argsort(new_edge_order,
                                 key=lambda e: self.edges.index(e))
        self.edge_colors = [self.edge_colors[i] for i in new_edge_order]

    def apply_inverse_vertex_map(self, inverse_vertex_map: List[int]) -> None:
        vertex_map = argsort(inverse_vertex_map)
        return self.apply_vertex_map(vertex_map)

    def subcoloring(
            self,
            indices: Union[List[int], Tuple[int]],
            as_vertices: bool = False):

        if as_vertices:
            vertices = indices
            assert all([v in vertices for v in range(self.ftype_size)])
            assert all([v in self.vertices for v in vertices])

        else:
            for i in indices:
                assert 0 <= i < self.nvertices - self.ftype_size
            vertices = list(range(self.ftype_size))
            vertices += [self.ftype_size+i for i in indices]

        sub_edges = get_edges(vertices)

        sub_edge_colors = []
        for edge in sub_edges:
            try:
                idx = self.edges.index(edge)
                color = self.edge_colors[idx]
            except ValueError:
                color = 0
            sub_edge_colors.append(color)

        return self.__class__(
            len(vertices),
            ncolors=self.ncolors,
            edge_colors=sub_edge_colors,
            ftype_size=self.ftype_size)

    def contains_subcoloring(
            self,
            subcoloring,
            color_invariant: Union[bool, Tuple[Tuple[int]]] = False,
            fixed_vertices: Optional[List[int]] = None,
            is_canonical: bool = False) -> bool:

        if subcoloring.nvertices > self.nvertices or subcoloring.ncolors != self.ncolors:
            return False

        if not is_canonical:
            subcoloring.canonize(color_invariant=color_invariant)

        fvertices = fixed_vertices or []
        assert subcoloring.nvertices >= len(fvertices)
        rvertices = [v for v in self.vertices if v not in fvertices]

        for T in combinations(rvertices, subcoloring.nvertices-len(fvertices)):
            T = list(T) + fvertices
            temp = self.subcoloring(T, as_vertices=True)
            temp.canonize(color_invariant=color_invariant)
            if temp == subcoloring:
                return True

        return False

    def _graph_representation(
            self,
            color_invariant: Union[bool, Tuple[Tuple[int]]] = False):

        if color_invariant is False:
            if hasattr(self, '_gr') and hasattr(self, '_pt'):
                return self._gr, self._pt

            graph_repr = Graph(self.nvertices)

            edges = []
            for (v1, v2), color in zip(self.edges, self.edge_colors):
                edges.append((v1, v2, color))
            graph_repr.add_edges(edges)

            pt = [[i] for i in range(self.ftype_size)]
            pt.append(list(range(self.ftype_size, self.nvertices)))

            self._gr, self._pt = graph_repr, pt

        elif color_invariant is True or isinstance(color_invariant, tuple):
            assert self.ftype_size == 0

            if hasattr(self, '_cgr') and hasattr(self, '_cpt'):
                return self._cgr, self._cpt

            n = self.nvertices
            graph_repr = Graph(n+self.ncolors+len(self))

            arguments = enumerate(zip(self.edges, self.edge_colors))
            for edge_idx, (edge, color) in arguments:
                nedge = (n+color, n+self.ncolors+edge_idx, None)
                graph_repr.add_edge(nedge)

                for v in edge:
                    nedge = (v, n+self.ncolors+edge_idx, None)
                    graph_repr.add_edge(nedge)

            pt = [[i] for i in range(self.ftype_size)]
            pt.append(list(range(self.ftype_size, n)))
            if color_invariant is True:
                pt.append(list(range(n, n+self.ncolors)))
            else:
                for color_part in color_invariant:
                    pt.append([n+c for c in color_part])
            pt.append(list(range(n+self.ncolors, n+self.ncolors+len(self))))

            self._cgr, self._cpt = graph_repr, pt

        return graph_repr, pt

    def canonical_representative(
            self, algorithm: str = 'sage',
            color_invariant: Union[bool, Tuple[Tuple[int]]] = False,
            certificate: bool = False):

        graph_representation, partition = self._graph_representation(
            color_invariant=color_invariant)

        _, cert = graph_representation.canonical_label(
            algorithm=algorithm,
            partition=partition,
            edge_labels=not color_invariant,
            certificate=True)

        canonical_representative = deepcopy(self)

        vertex_map = [cert[v] for v in self.vertices]
        canonical_representative.apply_vertex_map(vertex_map)

        color_map = None
        if color_invariant:
            n = self.nvertices
            color_map = [cert[n+c]-n for c in self.colors]
            canonical_representative.apply_color_map(color_map)

        if certificate:
            return canonical_representative, (vertex_map, color_map)

        return canonical_representative

    def canonize(
            self, algorithm: str = 'sage',
            color_invariant: Union[bool, Tuple[Tuple[int]]] = False,
            certificate: bool = False):

        output = self.canonical_representative(
            color_invariant=color_invariant,
            algorithm=algorithm,
            certificate=certificate)

        if certificate:
            cr, cert = output
        else:
            cr, cert = output, None

        self.__init__(
            cr.nvertices,
            ncolors=cr.ncolors,
            edge_colors=cr.edge_colors,
            ftype_size=cr.ftype_size)

        if certificate:
            return cert

    def automorphisms(
            self, algorithm='sage',
            color_invariant: Union[bool, Tuple[Tuple[int]]] = False):

        if color_invariant:
            if hasattr(self, '_cautom'):
                return self._cautom

        else:
            if hasattr(self, '_autom'):
                return self._autom

        graph_representation, partition = self._graph_representation(
            color_invariant=color_invariant)

        automorphisms = graph_representation.automorphism_group(
            algorithm=algorithm,
            partition=partition,
            edge_labels=not color_invariant)

        if color_invariant:
            self._cautom = automorphisms
        else:
            self._autom = automorphisms

        return automorphisms

    def make_ftype(
            self, color_invariant: Union[bool, Tuple[Tuple[int]]] = False):
        assert self._ftype_size == 0
        self.canonize(color_invariant=color_invariant)
        self._ftype_size = self.nvertices
        self._clear()

    def set_ftype(
            self, vertices: List[int],
            color_invariant: Union[bool, Tuple[Tuple[int]]] = False):

        assert self.ftype_size == 0

        ftype = self.subcoloring(vertices)
        cert = ftype.canonize(certificate=True,
                              color_invariant=color_invariant)
        ftype.make_ftype(color_invariant=color_invariant)

        vertex_map, count = [], 0
        for v in self.vertices:
            if v in vertices:
                vertex_map.append(cert[0][vertices.index(v)])
            else:
                vertex_map.append(len(vertices) + count)
                count += 1
        self.apply_vertex_map(vertex_map)

        if color_invariant:
            color_map = cert[1]
            self.apply_color_map(color_map)

        self._ftype_size = len(vertices)
        self._clear()

        assert self.ftype == ftype, (self.ftype, ftype)

    def apply_ftype_homomorphism(
            self, hom,
            color_invariant: Union[bool, Tuple[Tuple[int]]] = False) -> None:

        vertex_map = [hom(v) for v in range(self.ftype_size)]
        vertex_map += [v for v in range(self.ftype_size, self.nvertices)]
        self.apply_vertex_map(vertex_map)

        if color_invariant:
            n = self.ftype_size
            color_map = [hom(n+c)-n for c in self.colors]
            self.apply_color_map(color_map)

    def apply_homomorphism(
            self, hom,
            color_invariant: Union[bool, Tuple[Tuple[int]]] = False) -> None:

        vertex_map = [hom[v] for v in self.vertices]
        self.apply_vertex_map(vertex_map)

        if color_invariant:
            n = self.nvertices
            color_map = [hom(n+c)-n for c in self.colors]
            self.apply_color_map(color_map)


def get_canonical_extensions(
        coloring: Coloring,
        color_invariant: Union[bool, Tuple[Tuple[int]]],
        pre_select: bool,
        output_queue):
    """
    Generates all canonical extensions of a given coloring.

    Parameters:
    coloring (Coloring): The initial coloring.
    color_invariant (bool or Tuple[Tuple[int]]): Color invariant specification.
    pre_select (bool): Whether to pre-select candidates based on symmetry.
    output_queue: The queue to output the results.

    Returns:
    The canonical extensions of the given coloring.
    """

    output = []

    try:
        nvertices = coloring.nvertices

        min_edges = [combinations(coloring.vertices, 1)]
        min_edges = [tuple(sorted(edge)) for part in min_edges for edge in part]

        if pre_select:
            automorphisms = coloring.automorphisms(color_invariant=color_invariant)

            extension_candidates, seen = [], []

            for candidate in product(coloring.colors, repeat=len(min_edges)):
                if candidate in seen:
                    continue

                extension_candidates.append(candidate)

                orbit = []
                for autom in automorphisms:
                    autom = autom.dict()
                    vm = [autom[v] for v in coloring.vertices]
                    if not color_invariant:
                        cm = [c for c in coloring.colors]
                    else:
                        cm = [autom[coloring.nvertices+c]-coloring.nvertices
                              for c in coloring.colors]

                    c_min_edges = [combinations(vm, 1)]
                    c_min_edges = [tuple(sorted(edge)) for part in c_min_edges for edge in part]

                    idxs = argsort(c_min_edges, key=lambda e: min_edges.index(e))
                    autom_candidate = tuple([cm[candidate[i]] for i in idxs])

                    if autom_candidate not in orbit:
                        orbit.append(autom_candidate)

                seen += orbit
                seen = list(set(seen))

        else:
            extension_candidates = list(product(coloring.colors, repeat=len(min_edges)))

        edges = get_edges(list(range(nvertices+1)))

        for candidate in extension_candidates:
            edge_colors = []

            for edge in edges:
                if nvertices in edge:
                    e = tuple(sorted([v for v in edge if v != nvertices]))
                    idx = min_edges.index(e)
                    color = candidate[idx]
                else:
                    color = coloring.get_edge_color(edge)
                edge_colors.append(color)

            extended_coloring = Coloring(
                coloring.nvertices+1,
                coloring.ncolors,
                edge_colors=edge_colors,
                ftype_size=coloring.ftype_size)

            extended_coloring.canonize(color_invariant=color_invariant)
            if output_queue is None:
                output.append(str(extended_coloring))
            else:
                output_queue.put(str(extended_coloring))

        if output_queue is None:
            output.append(nvertices)
        else:
            output_queue.put(coloring.nvertices)

    except Exception as e:
        if output_queue is None:
            raise e
        output_queue.put((e, traceback.format_exc()))

    if output_queue is None:
        return output


def get_colorings(
        nvertices: int,
        ncolors: int = 2,
        color_invariant: Union[bool, Tuple[Tuple[int]]] = False,
        seeds: Optional[Union[
            Set[Coloring],
            List[Coloring]]] = None,
        pre_select: bool = True,
        mp: Optional[Tuple] = None,
        verbose: bool = True):
    """
    Generates all colorings for a given number of vertices and colors.

    Parameters:
    nvertices (int): The number of vertices.
    ncolors (int): The number of colors.
    color_invariant (bool or Tuple[Tuple[int]]): Color invariant specification.
    seeds (Set[Coloring] or List[Coloring], optional): Initial seed colorings.
    pre_select (bool): Whether to pre-select candidates based on symmetry.
    mp (Optional[Tuple]): Multiprocessing parameters.
    verbose (bool): Verbosity flag.

    Returns:
    A list of all possible colorings.
    """

    assert nvertices >= 0
    assert ncolors >= 1

    if nvertices == 0:
        return [Coloring(
                 nvertices,
                 ncolors=ncolors)]

    if seeds is None:
        seeds = [Coloring(
                 0,
                 ncolors=ncolors,)]

    for coloring in list(seeds):
        assert coloring.nvertices <= nvertices
        assert coloring.ncolors == ncolors

    colorings, seen = [], [str(coloring) for coloring in seeds]
    n_started_tasks, n_finished_tasks = {}, {}

    if mp is not None:
        manager, pool = mp

        def print_update(final: bool = False):
            if verbose:
                output = []
                for n in n_started_tasks.keys():
                    output.append((n_started_tasks[n], n_finished_tasks.get(n, 0)))

                print('Progress:', ' | '.join([f'{y}/{x}'
                      if y < x else str(x) for x, y in output]),
                      '| ' + str(len(colorings)) + ' '*10,
                      end=("\n" if final else "\r"))

        output_queue = manager.Queue()
        for coloring in seeds:
            pool.apply_async(get_canonical_extensions,
                             args=(coloring, color_invariant,
                                   pre_select, output_queue))
            n = coloring.nvertices
            n_started_tasks[n] = n_started_tasks.get(n, 0) + 1

        while any([n_finished_tasks.get(n, 0) < n_started_tasks[n]
                   for n in n_started_tasks.keys()]):

            while not output_queue.empty():
                coloring_str = output_queue.get()

                if isinstance(coloring_str, tuple):
                    e, tb = coloring_str
                    print(tb)
                    raise e

                if isinstance(coloring_str, int):
                    n = coloring_str
                    n_finished_tasks[n] = n_finished_tasks.get(n, 0) + 1
                    print_update()
                    continue

                if coloring_str in seen:
                    continue

                seen.append(coloring_str)

                coloring = Coloring.from_string(coloring_str)
                n = coloring.nvertices

                if n < nvertices:
                    pool.apply_async(get_canonical_extensions,
                                     args=(coloring, color_invariant, pre_select, output_queue))
                    n_started_tasks[n] = n_started_tasks.get(n, 0) + 1
                    print_update()
                    continue

                elif n == nvertices:
                    colorings.append(coloring)
                    print_update()

                else:
                    raise ValueError

        print_update(final=True)

    else:
        colorings = []

        while len(seeds) > 0:
            next_seeds = []
            for coloring in seeds:
                temp = get_canonical_extensions(
                        coloring, color_invariant, pre_select, None)
                curr_nvertices = temp[-1]
                temp = [Coloring.from_string(c)
                        for c in set(temp[:-1])]
                if curr_nvertices+1 == nvertices:
                    colorings += temp
                else:
                    next_seeds += temp
            seeds = list(set(next_seeds))

        colorings = list(set(colorings))
        
    return colorings


def get_monochromatic_triangle_density(
        coloring,
        color_invariant: Union[bool, Tuple[Tuple[int]]] = True):
    """
    Computes the density of monochromatic triangles in a given coloring.

    Parameters:
    coloring (Coloring or str): The coloring or its string representation.
    color_invariant (bool or Tuple[Tuple[int]]): Color invariant specification.

    Returns:
    The density of monochromatic triangles in the given coloring.
    """

    if isinstance(coloring, str):
        coloring = Coloring.from_string(coloring)

    monochromatic_triangles = []

    for c in coloring.colors:
        triangle = Coloring(3, ncolors=coloring.ncolors) #, edge_colors=[c, c, c])
        triangle.canonize(color_invariant=color_invariant)
        monochromatic_triangles.append(str(triangle))
        
    monochromatic_triangles = set(monochromatic_triangles)

    count, total = 0, 0

    for S in combinations(coloring.vertices, 3):
        H = coloring.subcoloring(S, as_vertices=True)
        H.canonize(color_invariant=color_invariant)

        total += 1

        if str(H) in monochromatic_triangles:
            count += 1

    return QQ(count / total)



def get_triangle_quadrangle_density(coloring):
    """
    Computes the density required for Theorem 2.3.

    Parameters:
    coloring (Coloring or str): The coloring or its string representation.
    color_invariant (bool or Tuple[Tuple[int]]): Color invariant specification.

    Returns:
    The density of monochromatic triangles and quadrangles in the given coloring.
    """

    if isinstance(coloring, str):
        coloring = Coloring.from_string(coloring)

    assert coloring.ncolors == 3


    triangle = Coloring(3, ncolors=coloring.ncolors, edge_colors=[0, 0, 0])
    triangle.canonize(color_invariant=((0, 1), (2,)))
    
    quadrangle = Coloring(4, ncolors=coloring.ncolors, edge_colors=[2, 2, 2, 2, 2, 2])
    quadrangle.canonize(color_invariant=((0, 1), (2,)))
    
    triangle_count, triangle_total = 0, 0
    for S in combinations(coloring.vertices, 3):
        H = coloring.subcoloring(S, as_vertices=True)
        H.canonize(color_invariant=((0, 1), (2,)))

        triangle_total += 1

        if H == triangle:
            triangle_count += 1

    quadruple_count, quadruple_total = 0, 0
    for S in combinations(coloring.vertices, 4):
        H = coloring.subcoloring(S, as_vertices=True)
        H.canonize(color_invariant=((0, 1), (2,)))

        quadruple_total += 1

        if H == triangle:
            quadruple_count += 1

    return QQ(triangle_total / triangle_total) + QQ(quadruple_count / quadruple_total)