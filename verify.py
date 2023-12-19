import os
import time
import yaml
import numpy as np

from collections import defaultdict
from itertools import combinations_with_replacement
from typing import Optional, List, Tuple, Dict, Any, Union
from sage.all import PermutationGroup, QQ, real, imag, matrix
from utilities import *

import sys
assert sys.version_info >= (3, 7), "Python version must be 3.7 or higher"


def get_orbits(
        automorphisms, flags: list,
        color_invariant: Union[bool, Tuple[Tuple[int]]]):
    """
    Calculate the orbits for a given set of automorphisms and flags.

    Args:
        automorphisms: Automorphism group of a given type.
        flags: List of flags of a given typoe.
        color_invariant: Indicates if color invariance is considered.

    Returns:
        A tuple containing orbits, pair orbits, flag group, and flag pair group.
    """
    permutations = []

    for a in list(automorphisms.gens()):
        permutation = [0 for _ in range(len(flags))]

        for i, flag in enumerate(flags):
            tflag = deepcopy(flag)

            tflag.apply_ftype_homomorphism(
                a, color_invariant=color_invariant)
            tflag.canonize(color_invariant=False)

            permutation[i] = flags.index(tflag)+1

        permutations.append(permutation)

    flag_group = PermutationGroup(permutations)

    order, orbits = int(flag_group.order()), []

    for orbit in flag_group.orbits():
        orbit = [flags[i-1] for i in orbit]
        assert order % len(orbit) == 0
        orbits.append((orbit, int(order / len(orbit))))

    pair_orbits, flag_pair_group = get_pair_orbits(flag_group, flags)

    return orbits, pair_orbits, flag_group, flag_pair_group


def get_pair_orbits(flag_group, flags: list):
    """
    Compute the orbits of flag pairs under the action of a flag group.

    Args:
        flag_group: The group of automorphisms acting on flags.
        flags: List of flags.

    Returns:
        A tuple containing pair orbits and flag pair group.
    """
    flag_indices = list(range(1, len(flags)+1))
    group_info = [[i-1 for i in a(flag_indices)] for a in flag_group]
    group_info = np.array(group_info).T

    seen, pair_orbits = {i: [] for i in range(len(flags))}, []

    for idx1, idx2 in combinations_with_replacement(range(len(flags)), 2):
        if idx2 in seen.get(idx1):
            continue

        orbit = list(zip(group_info[idx1], group_info[idx2]))
        orbit += [(j, i) for i, j in orbit]
        orbit = sorted(set(orbit))

        assert (idx1, idx2) in orbit

        for i, j in orbit:
            seen[i].append(j)

        assert 2*int(flag_group.order()) % len(orbit) == 0

        orbit = [(flags[i], flags[j]) for i, j in orbit]
        multiplicity = int(2*flag_group.order()/len(orbit))

        pair_orbits.append((orbit, multiplicity))

    return pair_orbits, None


def get_flowers(nvertices: int, ftype_orders: List[int]) -> Dict:
    """
    Generate flower structures for flags.

    Args:
        nvertices: Number of vertices in the coloring.
        ftype_orders: List of type orders.

    Returns:
        A dictionary of flower structures for each flag type order.
    """
    flowers = {t: [] for t in ftype_orders}

    for t in ftype_orders:
        f = get_max_flag_order(nvertices, t)
        for S1 in combinations(range(nvertices - t), f - t):
            S2 = tuple([i for i in range(nvertices - t) if i not in S1])
            if (S2, S1) not in flowers[t]:
                flowers[t].append((S1, S2))

    return flowers


def get_pair_densities(
        coloring,
        flags: dict,
        orbit_map: dict,
        flowers: dict,
        color_invariant: Union[bool, Tuple[Tuple[int]]] = False) -> dict:
    """
    Calculate the densities of flag pairs in given colorings.

    Args:
        coloring: The given coloring.
        flags: Dictionary of flags.
        orbit_map: Mapping of orbits.
        flowers: Flower structures.
        color_invariant: Indicates if color invariance is considered.

    Returns:
        A dictionary of flag pair densities.
    """
    
    n = coloring.nvertices
    flag_pair_densities = defaultdict(lambda: defaultdict(int))

    for t in set([ftype.nvertices for ftype in flags]):
        f = get_max_flag_order(n, t)

        denom = falling_factorial(n, t) * comb(n-t, f-t) * comb(n-f, f-t)

        for T in combinations(coloring.vertices, t):
            flag = deepcopy(coloring)
            flag.set_ftype(T, color_invariant=color_invariant)

            for S1, S2 in flowers[t]:
                flag1 = flag.subcoloring(S1)
                flag2 = flag.subcoloring(S2)

                flag1.canonize(color_invariant=False)
                flag2.canonize(color_invariant=False)

                oidx, mult = orbit_map[flag.ftype][(flag1, flag2)]

                flag_pair_densities[flag.ftype][oidx] += QQ(mult / denom)

                del flag1, flag2

    return {key: dict(value) for key, value in flag_pair_densities.items()}


def get_isotypic_diagonalization(
        flag_automorphisms: Any,
        nflags: int):
    """
    Get base change matrices for the isotypic diagonalization for given flag
    automorphisms.

    Args:
        flag_automorphisms: Automorphisms of the flags.
        nflags (int): Number of flags.

    Returns:
        A tuple containing base change matrices for the individual blocks
        resulting from the isotypic diagonalization.
    """

    def get_matrix_reps(a):
        M = matrix(nflags)
        for i, j in enumerate(range(nflags)):
            M[a(j+1)-1, i] = 1
        return M

    repms = {}
    for a in flag_automorphisms:
        repms[a] = get_matrix_reps(a)

    characters = {}
    conjugacy_classes = flag_automorphisms.conjugacy_classes()
    orthcharacter = [0 for _ in conjugacy_classes]

    for char in flag_automorphisms.character_table():
        char = tuple(char)
        testchar = tuple(real(c) for c in char)
        if any(c not in QQ for c in testchar):
            for i, tc in enumerate(testchar):
                orthcharacter[i] += tc
            continue
        if not all(imag(c) == 0 for c in char):
            char_real = tuple(2*real(c) for c in char)
        else:
            char_real = deepcopy(char)
        if char_real not in characters:
            characters[char_real] = {}
            for c, r, cclass in zip(char, char_real, conjugacy_classes):
                for a in cclass:
                    characters[char_real][a] = r

    bcmatrices = []

    for char in characters:
        dim = sum(characters[char][a.inverse()] * repms[a].trace()
                  for a in flag_automorphisms)
        if dim == 0:
            continue

        Mtemp = sum(characters[char][a.inverse()] * repms[a]
                    for a in flag_automorphisms)

        Mtemp, _ = Mtemp.T.gram_schmidt()
        Mtemp = Mtemp.T

        bcmatrices.append(matrix(QQ, Mtemp, sparse=True))

    return tuple(bcmatrices)
    


if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-dm', '--disable-multiprocessing',
                        action='store_true', help='disable multiprocessing')

    args = parser.parse_args()
    
    MP = None
    if not args.disable_multiprocessing:
        from multiprocessing import Manager, Pool
        MP = Manager(), Pool()
       
    NVERTICES = 6
    NCOLORINGS = {
        3: 3,
        4: 22,
        5: 513,
        6: 67685,
    }

    ########################
    # Determine all graphs #

    tm = time.perf_counter()
    print(f"\nDetermining the {NCOLORINGS[NVERTICES]} 4-edge-colorings of the complete graph of order {NVERTICES}...")

    colorings = get_colorings(NVERTICES, ncolors=4, color_invariant=True, mp=MP)
    colorings = sorted(colorings) # fix canonical order

    tm = time.perf_counter() - tm
    print(f"Found {len(colorings)} colorings in {tm:.1f}s.")


    #################################
    # Determine values of colorings #

    tm = time.perf_counter()
    print("\nDetermining values of colorings ...")

    results = apply_pool(get_monochromatic_triangle_density, colorings, mp=MP)
    values = {coloring: val for coloring, val in zip(colorings, results)}

    tm = time.perf_counter() - tm
    print(f"Done in {tm:.1f}s.")


    #######################
    # Determine all types #

    ftype_orders = [i for i in range(NVERTICES) if (NVERTICES-i) % 2 == 0]
    print(f"\nDetermining types of orders {', '.join([str(t) for t in ftype_orders])}...")

    tm = time.perf_counter()
    ftypes = {}
    for ftype_order in ftype_orders:
        for ftype in get_colorings(ftype_order, ncolors=4, color_invariant=True, mp=MP, verbose=False):
            automs = ftype.automorphisms(color_invariant=True)
            ftype.make_ftype(color_invariant=True)
            ftypes[ftype] = automs
    ftypes = {ftype: ftypes[ftype] for ftype in sorted(ftypes.keys())}
    tm = time.perf_counter() - tm

    print(f"Found {len(ftypes)} types in {tm:.1f}s.")


    #######################
    # Determine all flags #

    print("\nDetermining flags ...")

    tm = time.perf_counter()
    flags = {}
    for ftype in pbar(ftypes, leave=True):
        flags[ftype] = sorted(get_colorings(
            get_max_flag_order(NVERTICES, ftype.nvertices),
            seeds=[ftype],
            ncolors=4,
            color_invariant=False,
            mp=MP,
            verbose=False))
    tm = time.perf_counter() - tm

    print(f"Found {', '.join([str(len(f)) for f in flags.values()])} flags in {tm:.1f}s.")


    ################################
    # Determine flag (pair) orbits #

    print("\nDetermining flag (pair) orbits ...")

    tm = time.perf_counter()
    orbits, pair_orbits, flag_groups, flag_pair_groups = {}, {}, {}, {}

    arguments = [(automorphisms, flags[ftype], True)
                    for ftype, automorphisms in ftypes.items()]
    results = apply_pool(get_orbits, arguments, mp=MP, verbose=True)

    for ftype, (os, pos, fg, fpg) in zip(ftypes, results):
        orbits[ftype] = sorted([(sorted(o), m) for o, m in os])
        pair_orbits[ftype] = sorted([(sorted(o), m) for o, m in pos])
        flag_groups[ftype] = fg
        flag_pair_groups[ftype] = fpg

    tm = time.perf_counter() - tm
    print("Found", ', '.join([str(len(f)) for f in pair_orbits.values()]), f"pair orbits in {tm:.1f}s.")


    ###############################
    # Compute flag pair densities #

    print("\nComputing flag pair densities ...")

    tm = time.perf_counter()
    flowers = get_flowers(NVERTICES, ftype_orders)

    orbit_map = {}
    for ftype in ftypes:
        orbit_map[ftype] = {}
        for oidx, (orbit, mult) in enumerate(pair_orbits[ftype]):
            for idxs in orbit:
                orbit_map[ftype][idxs] = (oidx, mult)

    arguments = [(coloring, flags, orbit_map, flowers, True)
                    for coloring in colorings]
    results = apply_pool(get_pair_densities, arguments, mp=MP, verbose=True)
    pair_densities = {c: val for c, val in zip(colorings, results)}

    print(f"Done in {time.perf_counter() - tm:.1f}s")


    #######################################################################
    # Turn everything into string representation and give canonical order #

    tm = time.perf_counter()
    print("\nConverting to string representation and fixing order ...")

    orbits = {str(ftype): [([str(flag) for flag in orbit], m)
                for orbit, m in orbits[ftype]] for ftype in ftypes}
    pair_orbits = {str(ftype): [([(str(flag1), str(flag2))
                    for flag1, flag2 in orbit], m)
                    for orbit, m in pair_orbits[ftype]]
                    for ftype in ftypes}
    flags = {str(k): [str(flag) for flag in v] for k, v in flags.items()}

    pair_densities = {str(c): {str(ftype): pair_densities[c][ftype]
                        for ftype in ftypes if ftype in pair_densities[c]}
                        for c in colorings}
    values = {str(coloring): values[coloring] for coloring in colorings}
    colorings = [str(coloring) for coloring in colorings]

    ftypes = {str(ftype): v for ftype, v in ftypes.items()}
    flag_groups = {str(ftype): v for ftype, v in flag_groups.items()}
    flag_pair_groups = {str(ftype): v for ftype, v in flag_pair_groups.items()}

    tm = time.perf_counter() - tm
    print(f"Done in {tm:.1f}s.")
    
    
    #######################
    # Loading certificate #

    assert NVERTICES == 6, "The certificate is for N=6"

    tm = time.perf_counter()
    print(f"\nLoading the certificate...")

    with open('certificate_22.yaml', 'r') as file:
        loaded_certificate = yaml.safe_load(file)

    tm = time.perf_counter() - tm
    print(f"Done in in {tm:.1f}s.")


    ##########################
    # Processing certificate #

    tm = time.perf_counter()
    print(f"\nProcessing the certificate...")

    total_iterations = sum(len(v) for v in loaded_certificate.values())
    progressbar = get_pbar(total=int(total_iterations))

    certificate = {k: {} for k in loaded_certificate.keys()}

    for ftype, v in loaded_certificate.items():
        for flag1, flag2, value in v:
            index = next((i for i, orbit in enumerate(pair_orbits[ftype]) if (flag1, flag2) in orbit[0]), -1)
            assert 0 <= index < len(pair_orbits[ftype])
            certificate[ftype][index] = QQ(value)

            progressbar.update(int(1))

    progressbar.close()

    tm = time.perf_counter() - tm
    print(f"Done in in {tm:.1f}s.")
    
    
    ##############################
    # Veriyfing constraint slack #
    
    print(f"\nVerifying slack conditions...")
    
    K33 = Coloring(6, ncolors=4, edge_colors=[0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2])
    K33.canonize(color_invariant=True)

    K31_1 = Coloring(4, ncolors=4, edge_colors=[0, 0, 0, 0, 0, 1])
    K31_1.canonize(color_invariant=True)

    K31_2 = Coloring(4, ncolors=4, edge_colors=[0, 0, 0, 0, 2, 1])
    K31_2.canonize(color_invariant=True)

    K31_3 = Coloring(4, ncolors=4, edge_colors=[0, 0, 2, 0, 2, 1])
    K31_3.canonize(color_invariant=True)

    for coloring in pbar(colorings):
        lam = values[coloring]
        for ftype in certificate.keys():
            pd = pair_densities[coloring].get(ftype, {})
            orbits = pair_orbits[ftype]
            for oidx, v in pd.items():
                lam -= v * certificate[ftype].get(oidx, 0) * len(orbits[oidx][0])
                
        # Checking that 1/256 is the correct value
        assert lam >= QQ(1/256), (lam, float(lam))
        
        if lam == QQ(1/256):
            coloring = Coloring.from_string(coloring)
        
            # Checking that any coloring attaining that value does not contain a forbidden subgraph
            assert not coloring.contains_subcoloring(K33, color_invariant=True, is_canonical=True)
            assert not coloring.contains_subcoloring(K31_1, color_invariant=True, is_canonical=True)
            assert not coloring.contains_subcoloring(K31_2, color_invariant=True, is_canonical=True)
            assert not coloring.contains_subcoloring(K31_3, color_invariant=True, is_canonical=True)
            
    
    #######################################
    # Veriyfing positive semidefiniteness #
    
    print(f"\nVerifying positive semidefinites...")

    for ftype, vals in certificate.items():
        
        flag_group = flag_groups[ftype]
        nflags = len(flags[ftype])
        
        print(f"\n  - creating {nflags}x{nflags} Q-matrix for {ftype}")
        Q = matrix(QQ, nflags, nflags, sparse=True)

        for oidx, value in vals.items():
            orbit, _ = pair_orbits[ftype][oidx]
            for i, j in orbit:
                Q[flags[ftype].index(i), flags[ftype].index(j)] = value
        
        tm = time.perf_counter()
        print(f"  - getting diagonalization matrices for {ftype}")
        diag_matrices = get_isotypic_diagonalization(flag_group, nflags)
        tm = time.perf_counter() - tm
        print(f"    done in in {tm:.1f}s")

        print(f"  - verifying positive-semidefinitess for {ftype}")
        for BC in diag_matrices:
            tm = time.perf_counter()
            Qd = matrix(QQ, BC.T * Q * BC, sparse=True)
            print (f"     * checking {Qd.nrows()}x{Qd.ncols()} block")
            
            # Verifying positive semidefinites *exactly* and not numerically
            evs = Qd.eigenvalues()
            assert all([v >= 0 for v in evs])
            
            tm = time.perf_counter() - tm
            print(f"       found {sum([v == 0 for v in evs])} zero eigenvalues, no negative ones")
            print(f"       done in in {tm:.1f}s")
            
    
    if MP is not None:
        _, pool = MP
        pool.close()