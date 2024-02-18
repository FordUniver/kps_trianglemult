import os
import time
import yaml
import pickle
import numpy as np

from collections import defaultdict
from itertools import combinations_with_replacement
from typing import Optional, List, Tuple, Dict, Any, Union
from sage.all import PermutationGroup, QQ, RR, real, imag, matrix, vector
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
    
    while True:
        theorem = input("\nWhich theorem do you want to verify (2.3 or 2.4)? ")
        if theorem in ["2.3", "2.4", "cummingsetal", "goodman"]:
            break
        print("Invalid input. Please enter 2.3 or 2.4.")
    
    if theorem == '2.3':
        NVERTICES = 6
        NCOLORINGS = {
            3: 3,
            4: 22,
            5: 513,
            6: 67685,
        }
        NCOLORS = 4
        CINV = True
        
        K33 = Coloring(6, ncolors=4, edge_colors=[0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2])
        K33.canonize(color_invariant=CINV)

        K31_1 = Coloring(4, ncolors=4, edge_colors=[0, 0, 0, 0, 0, 1])
        K31_1.canonize(color_invariant=CINV)

        K31_2 = Coloring(4, ncolors=4, edge_colors=[0, 0, 0, 0, 2, 1])
        K31_2.canonize(color_invariant=CINV)

        K31_3 = Coloring(4, ncolors=4, edge_colors=[0, 0, 2, 0, 2, 1])
        K31_3.canonize(color_invariant=CINV)
        
        forbidden = [K33, K31_1, K31_2, K31_3]
        target = QQ(1/256)
        
    elif theorem == '2.4':
        NVERTICES = 6
        NCOLORINGS = {5: 142, 6: 12796 }
        NCOLORS = 3
        CINV = ((0, 1), (2, ))
        
        forbidden = []
        target = QQ(1/125)
        
    elif theorem == 'cummingsetal':
        NVERTICES = 5
        NCOLORINGS = { 5: 142, 6: 12796 }
        NCOLORS = 3
        CINV = True
        
        K33 = Coloring(6, ncolors=3, edge_colors=[0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2])
        K33.canonize(color_invariant=CINV)

        K31_1 = Coloring(4, ncolors=3, edge_colors=[0, 0, 0, 0, 0, 1])
        K31_1.canonize(color_invariant=CINV)

        K31_2 = Coloring(4, ncolors=3, edge_colors=[0, 0, 0, 0, 2, 1])
        K31_2.canonize(color_invariant=CINV)

        K31_3 = Coloring(4, ncolors=3, edge_colors=[0, 0, 2, 0, 2, 1])
        K31_3.canonize(color_invariant=CINV)
        
        forbidden = [K33, K31_1, K31_2, K31_3]
        
        target = QQ(1/25)
        
    elif theorem == 'goodman':
        NVERTICES = 3
        NCOLORINGS = { 3: 2 }
        NCOLORS = 2
        CINV = True
        
        forbidden = []
        
        target = QQ(1/4)
        
    else:
        raise ValueError


    ########################
    # Determine all graphs #

    tm = time.perf_counter()
    print(f"\nDetermining the {NCOLORINGS[NVERTICES]} {NCOLORS}-edge-colorings of the complete graph of order {NVERTICES}...")

    colorings = get_colorings(NVERTICES, ncolors=NCOLORS, color_invariant=CINV, mp=MP)
    colorings = sorted(colorings) # fix canonical order

    tm = time.perf_counter() - tm
    print(f"Found {len(colorings)} colorings in {tm:.1f}s.")


    #################################
    # Determine values of colorings #

    tm = time.perf_counter()
    print("\nDetermining values of colorings ...")


    if theorem in ['2.3', 'cummingsetal', 'goodman']:
        results = apply_pool(get_monochromatic_triangle_density, colorings, mp=MP)
        values = {coloring: val for coloring, val in zip(colorings, results)}
        
    elif theorem == '2.4':
        results = apply_pool(get_triangle_quadrangle_density, colorings, mp=MP)
        values = {coloring: val for coloring, val in zip(colorings, results)}
        
    else:
        raise ValueError
    

    tm = time.perf_counter() - tm
    print(f"Done in {tm:.1f}s.")


    #######################
    # Determine all types #

    ftype_orders = [i for i in range(NVERTICES) if (NVERTICES-i) % 2 == 0]
    print(f"\nDetermining types of orders {', '.join([str(t) for t in ftype_orders])}...")

    tm = time.perf_counter()
    ftypes = {}
    for ftype_order in ftype_orders:
        for ftype in get_colorings(ftype_order, ncolors=NCOLORS, color_invariant=CINV, mp=MP, verbose=False):
            automs = ftype.automorphisms(color_invariant=CINV)
            ftype.make_ftype(color_invariant=CINV)
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
            ncolors=NCOLORS,
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

    for ftype, (obs, pos, fg, fpg) in zip(ftypes, results):
        orbits[ftype] = sorted([(sorted(o), m) for o, m in obs])
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

    arguments = [(coloring, flags, orbit_map, flowers, CINV)
                    for coloring in colorings]
    results = apply_pool(get_pair_densities, arguments, mp=MP, verbose=True)
    pair_densities = {c: val for c, val in zip(colorings, results)}


    # Turn everything into string representation and give canonical order 

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
        

    ##########################
    # Processing certificate #

    tm = time.perf_counter()
    print(f"\nProcessing the certificate...")

    with open(f"certificates/certificate_{theorem.replace('.','')}.yaml", 'r') as file:
        loaded_certificate = yaml.safe_load(file)

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
    # Verifying constraint slack #
    
    print(f"\nVerifying slack conditions...")

    found_sharp = False

    for coloring in pbar(colorings):
        lam = values[coloring]
        for ftype in certificate.keys():
            pd = pair_densities[coloring].get(ftype, {})
            orbits = pair_orbits[ftype]
            for oidx, v in pd.items():
                lam -= v * certificate[ftype].get(oidx, 0) * len(orbits[oidx][0])
                
        # Checking that target is the correct value
        assert lam >= target, (lam, float(lam))
        
        found_sharp = found_sharp or lam == target
        
        if lam == target:
            coloring = Coloring.from_string(coloring)
        
            # Checking that any coloring attaining that value does not contain a forbidden subgraph
            for H in forbidden:
                assert not coloring.contains_subcoloring(H, color_invariant=CINV, is_canonical=True)
    
    assert found_sharp
    
    print(f"The value is {target} and the correct graphs are sharp ✅")
    
    
    ####################################
    # Verifying known zero eigenvalues #
    
    tm = time.perf_counter()
    print(f"\nVerifying known zero_eigenvalues...")
    
    Q_matrices = {}

    for ftype, vals in certificate.items():
        flag_group = flag_groups[ftype]
        nflags = len(flags[ftype])
        
        tm = time.perf_counter()
        Q_matrices[ftype] = matrix(QQ, nflags, nflags, sparse=True)
    
        for oidx, value in vals.items():
            orbit, _ = pair_orbits[ftype][oidx]
            for i, j in orbit:
                 Q_matrices[ftype][flags[ftype].index(i), flags[ftype].index(j)] = value
        
    with open(f"certificates/known_zevs_{theorem.replace('.','')}.yaml", 'r') as file:
        temp = yaml.safe_load(file)
       
    known_zevs = {str(k): [vector([QQ(x) for x in zev]) for zev in v] for k, v in temp.items()} 
    
    for ftype, vals in certificate.items():
        zevs = known_zevs.get(ftype, [])
        nflags = len(flags[ftype])
        
        zev_matrix = matrix(QQ, zevs)
        
        assert zev_matrix.rank() == zev_matrix.nrows() == len(zevs)
        
        for zev in zevs:
            assert Q_matrices[ftype]*zev == vector([0 for _ in range(nflags)])
            assert zev*Q_matrices[ftype]*zev == 0
            
        print(f"  - {ftype} has {len(zevs)} known linearly independent zero eigenvalue(s) ✅")

    tm = time.perf_counter() - tm
    # print(f"Done in {tm:.1f}s.")
    
    
    ###################################################
    # Verifying positive semidefiniteness numerically #
    
    tolerance = 1e-6
    
    print(f"\nVerifying positive semidefinites numerically up to a tolerance of {tolerance} ...")
    tm = time.perf_counter()

    for ftype, vals in certificate.items():
        zevs = known_zevs.get(ftype, [])
        flag_group = flag_groups[ftype]
        nflags = len(flags[ftype])
        
        evs = np.linalg.eigvals(Q_matrices[ftype])
        
        nzev = sum([v <= tolerance for v in evs])
        if nzev == len(zevs):
            print(f"  - {ftype} has {len(zevs)} zero eigenvalue(s) ✅")
        else:
            print(f"  - {ftype} has {nzev} zero eigenvalue(s) (should be {len(zevs)}) ❌")
        
        
    tm = time.perf_counter() - tm
    # print(f"Done in in {tm:.1f}s")

    #######################################
    # Verifying positive semidefiniteness #
    
    print(f"\nVerifying positive semidefinites algebraically...\n")
    
    for ftype, vals in certificate.items():
        tm = time.perf_counter()
        
        # if ftype not in ldl_decomp.keys():
        flag_group = flag_groups[ftype]
        nflags = len(flags[ftype])
        
        print(f"Checking block for type {ftype}")
        print(f"  - getting diagonalization matrices")
        diag_matrices = get_isotypic_diagonalization(flag_group, nflags)
        tm = time.perf_counter() - tm
        print(f"     in {tm:.1f}s")

        nzev = 0

        print(f"  - verifying positive-semidefinitess")
        for idx, BC in enumerate(diag_matrices):
            tm = time.perf_counter()
            Qd = matrix(QQ, BC.T * Q_matrices[ftype] * BC, sparse=True)
            
            decomp_file = f"certificates/decomp/{theorem.replace('.','')}/{ftype}_{idx}.yaml"
            
            if not os.path.isfile(decomp_file):
                print (f"     * computing LDL decomp of {Qd.nrows()}x{Qd.ncols()} block")
                P, L, D = Qd.block_ldlt()
                P_list = [[i, j, str(P[i, j])] for i, j in P.nonzero_positions()]
                L_list = [[i, j, str(L[i, j])] for i, j in L.nonzero_positions()]
                D_list = [[i, j, str(D[i, j])] for i, j in D.nonzero_positions()]
                os.makedirs(f"certificates/decomp/{theorem.replace('.','')}", exist_ok=True)
                with open(decomp_file, 'w') as file:
                    yaml.dump([P_list, L_list, D_list], file, default_flow_style=True)
                    
            else:
                print (f"     * loading LDL decomp of {Qd.nrows()}x{Qd.ncols()} block")
                with open(decomp_file, 'r') as file:
                    P_list, L_list, D_list = yaml.safe_load(file)

                P = matrix(QQ, Qd.nrows(), Qd.ncols(), sparse=True)
                L = matrix(QQ, Qd.nrows(), Qd.ncols(), sparse=True)
                D = matrix(QQ, Qd.nrows(), Qd.ncols(), sparse=True)

                for i, j, val in P_list:
                    P[i, j] = QQ(val)
                for i, j, val in L_list:
                    L[i, j] = QQ(val)
                for i, j, val in D_list:
                    D[i, j] = QQ(val)
                
            assert P.T*Qd*P == L*D*L.T
            assert all(d >= 0 for d in D.diagonal()), "       block is not positive semidefinite ❌"
            
            nzev += sum([d == 0 for d in D.diagonal()])
            
            print(f"       block is positive semidefinite ✅")
            
            tm = time.perf_counter() - tm
            print(f"        in {tm:.1f}s")
            
        assert nzev == len(known_zevs.get(ftype, [])), f"       Found {nzev} zero eigenvalue(s) (should be {len(known_zevs.get(ftype, []))}) ❌"
        print(f"    found exactly {nzev} zero eigenvalue(s) ✅\n")
            
    if MP is not None:
        _, pool = MP
        pool.close()