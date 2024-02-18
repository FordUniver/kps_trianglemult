# The Four-Color Ramsey Multiplicity of Triangles
This repository contains the code accompanying the paper "The Four-Color Ramsey Multiplicity of Triangles" by Aldo Kiem, Sebastian Pokutta, and Christoph Spiegel.

## Description of the Certificate
The certificate `certificate_23.yaml` contains the rational entries of the solution matrices, one for each of the 22 possible types. They are stored in the format `{t: [[F1, F2, q], ... ], ...}` where t is a type and F1 and F2 are two flags over t and q is the rational entry. The matrices of the certificate are not diagonalized. This is because the entries in the diagonalized form are more complicated. Rounding the floating point solution was carried out too without the diagonalization. For each type t there is a color-invariant automorphism group and the entries of the solution matrices are constant on its orbits. Therefore, the certificate lists only the value q of one representative (F1, F2) of the orbit.

The types and flags are encoded with a string representation. As an example, consider `c.2.4.5.4.9k5e`. The prefix `c` indicates that the object is a coloring of a complete graph. In succession, 2 denotes the uniformity, 4 the number of colors, 5 the number of vertices and 4 that the first four vertices of the coloring form a type. The suffix `9k5e` is a base 36 string representation, using all digits and letters, of the list of edge colors. 

The certificate is verified as follows. We first need to generate for every coloring on six vertices G, the constraint matrices D for the SDP corresponding to G. Then, we take the Frobenius product X.D of our solution X with D. We need to verify that the monochromatic triangle density in G is greater than or equal to 1/256 + X.D. Secondly, we need to verify that each solution matrix X is semidefinite. This is achieved by computing all eigenvalues of X and making sure that its values are always at least 0. When these two conditions are met, we have a valid certificate for the lower bound 1/256. To get stability, we need to check that the triangle density in G is strictly greater than 1/256 + X.D for any coloring G on six vertices with a non-zero K31 or K33 density.

## Verification using verify.py
A script that makes the verification is provided in `verify.py`. It requires python version 3.7 or higher as well as an installation of sage, the python libraries tqdm and numpy and a yaml parser. To run the verification, type `python verify.py` in the terminal. An optional flag is `-dm` that disables multiprocessing.
