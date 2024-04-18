# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Constants used in AlphaFold."""

import collections
import functools
import os
from typing import List, Mapping, Tuple

import numpy as np
import tree

# Internal import (35fd).


# Distance from one CA to next CA [trans configuration: omega = 180].
c4_c4 = 6.2

# Format: The list for each AA type contains chi1, chi2, chi3, chi4 in
# this order (or a relevant subset from chi1 onwards). ALA and GLY don't have
# chi angles so their chi angle lists are empty.
chi_angles_atoms = {
    'A': [['O4\'', 'C1\'', 'N9', 'C4']],
    'G': [['O4\'', 'C1\'', 'N9', 'C4']],
    'C': [['O4\'', 'C1\'', 'N1', 'C2']],
    'U': [['O4\'', 'C1\'', 'N1', 'C2']],
}

# If chi angles given in fixed-length array, this matrix determines how to mask
# them for each AA type. The order is as per restype_order (see below).
chi_angles_mask = [
    [0.0, 0.0, 0.0, 0.0],  # ALA
    [1.0, 1.0, 1.0, 1.0],  # ARG
    [1.0, 1.0, 0.0, 0.0],  # ASN
    [1.0, 1.0, 0.0, 0.0],  # ASP
    [1.0, 0.0, 0.0, 0.0],  # CYS
    [1.0, 1.0, 1.0, 0.0],  # GLN
    [1.0, 1.0, 1.0, 0.0],  # GLU
    [0.0, 0.0, 0.0, 0.0],  # GLY
    [1.0, 1.0, 0.0, 0.0],  # HIS
    [1.0, 1.0, 0.0, 0.0],  # ILE
    [1.0, 1.0, 0.0, 0.0],  # LEU
    [1.0, 1.0, 1.0, 1.0],  # LYS
    [1.0, 1.0, 1.0, 0.0],  # MET
    [1.0, 1.0, 0.0, 0.0],  # PHE
    [1.0, 1.0, 0.0, 0.0],  # PRO
    [1.0, 0.0, 0.0, 0.0],  # SER
    [1.0, 0.0, 0.0, 0.0],  # THR
    [1.0, 1.0, 0.0, 0.0],  # TRP
    [1.0, 1.0, 0.0, 0.0],  # TYR
    [1.0, 0.0, 0.0, 0.0],  # VAL
]

# The following chi angles are pi periodic: they can be rotated by a multiple
# of pi without affecting the structure.
chi_pi_periodic = [
    [0.0, 0.0, 0.0, 0.0],  # ALA
    [0.0, 0.0, 0.0, 0.0],  # ARG
    [0.0, 0.0, 0.0, 0.0],  # ASN
    [0.0, 1.0, 0.0, 0.0],  # ASP
    [0.0, 0.0, 0.0, 0.0],  # CYS
    [0.0, 0.0, 0.0, 0.0],  # GLN
    [0.0, 0.0, 1.0, 0.0],  # GLU
    [0.0, 0.0, 0.0, 0.0],  # GLY
    [0.0, 0.0, 0.0, 0.0],  # HIS
    [0.0, 0.0, 0.0, 0.0],  # ILE
    [0.0, 0.0, 0.0, 0.0],  # LEU
    [0.0, 0.0, 0.0, 0.0],  # LYS
    [0.0, 0.0, 0.0, 0.0],  # MET
    [0.0, 1.0, 0.0, 0.0],  # PHE
    [0.0, 0.0, 0.0, 0.0],  # PRO
    [0.0, 0.0, 0.0, 0.0],  # SER
    [0.0, 0.0, 0.0, 0.0],  # THR
    [0.0, 0.0, 0.0, 0.0],  # TRP
    [0.0, 1.0, 0.0, 0.0],  # TYR
    [0.0, 0.0, 0.0, 0.0],  # VAL
    [0.0, 0.0, 0.0, 0.0],  # UNK
]

# Atoms positions relative to the 8 rigid groups, defined by the pre-omega, phi,
# psi and chi angles:
# 0: 'backbone group',
# 1: 'pre-omega-group', (empty)
# 2: 'phi-group', (currently empty, because it defines only hydrogens)
# 3: 'psi-group',
# 4,5,6,7: 'chi1,2,3,4-group'
# The atom positions are relative to the axis-end-atom of the corresponding
# rotation axis. The x-axis is in direction of the rotation axis, and the y-axis
# is defined such that the dihedral-angle-definiting atom (the last entry in
# chi_angles_atoms above) is in the xy-plane (with a positive y-coordinate).
# format: [atomname, group_idx, rel_position]
rigid_group_atom_positions = {
    'G': [
        ["P", 0, (-1.2650003, 1.092, -3.552)],
        ["OP1", 0, (-2.269001, 2.099, -3.124)],
        ["OP2", 0, (-0.32700062, 1.416, -4.657)],
        ["O5'", 0, (-0.40299988, 0.67999995, -2.277)],
        ["C5'", 0, (-1.026, 0.445, -1.0159998)],
        ["C4'", 0, (0.0, 0.0, 0.0)],
        ["O4'", 4, (0.69199944, -1.1760001, -0.5029998)],
        ["C1'", 4, (2.0509996, -1.1370001, -0.10099983)],
        ["N9", 4, (2.8859997, -1.1140001, -1.2979999)],
        ["C4", 4, (4.224, -1.4200001, -1.3650002)],
        ["N3", 2, (5.0039997, -1.7950001, -0.32999992)],
        ["C2", 2, (6.2489996, -2.023, -0.7090001)],
        ["N2", 2, (7.1559997, -2.408, 0.19199991)],
        ["N1", 2, (6.6959996, -1.891, -1.9980001)],
        ["C6", 2, (5.913, -1.506, -3.08)],
        ["O6", 2, (6.4189997, -1.417, -4.204)],
        ["C5", 2, (4.5699997, -1.2590001, -2.69)],
        ["N7", 2, (3.4729996, -0.85800004, -3.441)],
        ["C8", 2, (2.5009995, -0.78400004, -2.574)],
        ["C2'", 2, (2.2319994, 0.08899999, 0.7949996)],
        ["O2'", 2, (2.005, -0.29200003, 2.1390004)],
        ["C3'", 0, (1.118, 0.9929999, 0.28399992)],
        ["O3'", 0, (0.724, 1.9699999, 1.244)],
    ],
    'A':[
      ["P", 0, (0.717, -2.4960003, 2.822)],
      ["OP1", 0, (-0.6149998, -3.0090008, 3.2319999)],
      ["OP2", 0, (1.8959999, -3.3980007, 2.8639998)],
      ["O5'", 0, (0.592, -1.9090004, 1.3459997)],
      ["C5'", 0, (-0.4790001, -1.0359993, 0.9909997)],
      ["C4'", 0, (0.0, 0.0, 0.0)],
      ["O4'", 4, (0.71100044, 1.0430002, 0.71899986)],
      ["C1'", 4, (1.816, 1.4860001, -0.05000019)],
      ["N9", 4, (3.0410004, 1.2089996, 0.70099974)],
      ["C4", 4, (4.3129997, 1.5839996, 0.342)],
      ["N3", 2, (4.6809998, 2.2679996, -0.7550001)],
      ["C2", 2, (5.999, 2.4519997, -0.77300024)],
      ["N1", 2, (6.921, 2.0599995, 0.11499977)],
      ["C6", 2, (6.518, 1.375, 1.2059999)],
      ["N6", 2, (7.438, 0.98600006, 2.0879998)],
      ["C5", 2, (5.142, 1.1129999, 1.3429999)],
      ["N7", 2, (4.408, 0.4510002, 2.3179998)],
      ["C8", 2, (3.1710005, 0.53600025, 1.8909998)],
      ["C2'", 2, (1.7519999, 0.776, -1.402)],
      ["O2'", 2, (0.9989996, 1.5739994, -2.2970004)],
      ["C3'", 0, (1.0, -0.49699974, -1.0350003)],
      ["O3'", 0, (0.34500027, -1.085001, -2.1559997)],

    ],
    'C': [
        ["P", 0, (1.17, -0.28399992, -3.737)],
        ["OP1", 0, (2.257, -1.283, -3.586)],
        ["OP2", 0, (0.204, -0.41400003, -4.858)],
        ["O5'", 0, (0.346, -0.23200011, -2.374)],
        ["C5'", 0, (1.009, -0.052000046, -1.1240001)],
        ["C4'", 0, (0.0, 0.0, 0.0)],
        ["O4'", 4, (-0.773, 1.2279999, -0.109000206)],
        ["C1'", 4, (-2.1209998, 0.977, 0.2550001)],
        ["N1", 4, (-2.974, 1.2160001, -0.921)],
        ["C6", 2, (-2.452, 1.188, -2.184)],
        ["C2", 4, (-4.334, 1.4719999, -0.72599983)],
        ["O2", 2, (-4.782, 1.497, 0.4289999)],
        ["N3", 2, (-5.1270003, 1.688, -1.7990003)],
        ["C4", 2, (-4.6080003, 1.6539999, -3.027)],
        ["N4", 2, (-5.4300003, 1.865, -4.054)],
        ["C5", 2, (-3.225, 1.401, -3.256)],
        ["C2'", 2, (-2.188, -0.45799994, 0.77699995)],
        ["O2'", 2, (-1.9140002, -0.44799995, 2.1660004)],
        ["C3'", 0, (-1.0500001, -1.1009998, -0.004000187)],
        ["O3'", 0, (-0.57000005, -2.295, 0.6069999)],
    ],
    'U': [
        ["P", 0, (3.795, -0.89900017, 0.44699955)],
        ["OP1", 0, (3.6750002, -1.8889999, 1.5480003)],
        ["OP2", 0, (4.6780005, -1.1929998, -0.71000004)],
        ["O5'", 0, (2.3330002, -0.59000015, -0.10599995)],
        ["C5'", 0, (1.2550001, -0.3029995, 0.783)],
        ["C4'", 0, (0.0, 0.0, 0.0)],
        ["O4'", 4, (0.1760006, 1.2630005, -0.697)],
        ["C1'", 4, (-0.40999985, 1.1859999, -1.9840002)],
        ["N1", 4, (0.651, 1.3710003, -2.9850001)],
        ["C6", 2, (1.901, 0.8199997, -2.8120003)],
        ["C2", 4, (0.35300064, 2.119, -4.11)],
        ["O2", 2, (-0.73899937, 2.6220002, -4.302)],
        ["N3", 2, (1.3830004, 2.257, -5.004)],
        ["C4", 2, (2.6520004, 1.737, -4.894)],
        ["O4", 2, (3.4690003, 1.9440002, -5.79)],
        ["C5", 2, (2.8820004, 0.97599983, -3.706)],
        ["C2'", 2, (-1.1309996, -0.15799999, -2.0880003)],
        ["O2'", 2, (-2.4689999, 0.012000084, -1.6610003)],
        ["C3'", 0, (-0.34000015, -1.0019999, -1.0939999)],
        ["O3'", 0, (-1.0999994, -2.092, -0.5760002)],
    ]
    
}

# A list of atoms (excluding hydrogen) for each AA type. PDB naming convention.
# residue_atoms = {
#     'ALA': ['C', 'CA', 'CB', 'N', 'O'],
#     'ARG': ['C', 'CA', 'CB', 'CG', 'CD', 'CZ', 'N', 'NE', 'O', 'NH1', 'NH2'],
#     'ASP': ['C', 'CA', 'CB', 'CG', 'N', 'O', 'OD1', 'OD2'],
#     'ASN': ['C', 'CA', 'CB', 'CG', 'N', 'ND2', 'O', 'OD1'],
# }

residue_atoms = {
  'A': ["P", "OP1", "OP2", "O5\'", "C5\'", "C4\'", "O4\'", "C1\'", "N9", "C4", "N3", "C2", "N1", "C6", "N6", "C5", "N7", "C8", "C2\'", "O2\'", "C3\'", "O3\'"],
  'C': ["P", "OP1", "OP2", "O5\'", "C5\'", "C4\'", "O4\'", "C1\'", "N1", "C6", "C2", "O2", "N3", "C4", "N4", "C5", "C2\'", "O2\'", "C3\'", "O3\'"],
  'G': ["P", "OP1", "OP2",  "O5\'", "C5\'", "C4\'", "O4\'", "C1\'", "N9", "C4", "N3", "C2", "N2", "N1", "C6", "O6", "C5", "N7", "C8", "C2\'", "O2\'", "C3\'", "O3\'"],
  'U': ["P", "OP1", "OP2", "O5\'", "C5\'", "C4\'", "O4\'", "C1\'", "N1", "C6", "C2", "O2", "N3", "C4", "O4", "C5", "C2\'", "O2\'", "C3\'", "O3\'"]
}

# Naming swaps for ambiguous atom names.
# Due to symmetries in the amino acids the naming of atoms is ambiguous in
# 4 of the 20 amino acids.
# (The LDDT paper lists 7 amino acids as ambiguous, but the naming ambiguities
# in LEU, VAL and ARG can be resolved by using the 3d constellations of
# the 'ambiguous' atoms and their neighbours)
residue_atom_renaming_swaps = {
    'ASP': {'OD1': 'OD2'},
    'GLU': {'OE1': 'OE2'},
    'PHE': {'CD1': 'CD2', 'CE1': 'CE2'},
    'TYR': {'CD1': 'CD2', 'CE1': 'CE2'},
}

# Van der Waals radii [Angstroem] of the atoms (from Wikipedia)
van_der_waals_radius = {
    'C': 1.7,
    'N': 1.55,
    'O': 1.52,
    'S': 1.8,
}

Bond = collections.namedtuple(
    'Bond', ['atom1_name', 'atom2_name', 'length', 'stddev'])
BondAngle = collections.namedtuple(
    'BondAngle',
    ['atom1_name', 'atom2_name', 'atom3name', 'angle_rad', 'stddev'])


@functools.lru_cache(maxsize=None)
def load_stereo_chemical_props() -> Tuple[Mapping[str, List[Bond]],
                                          Mapping[str, List[Bond]],
                                          Mapping[str, List[BondAngle]]]:
  """Load stereo_chemical_props.txt into a nice structure.

  Load literature values for bond lengths and bond angles and translate
  bond angles into the length of the opposite edge of the triangle
  ("residue_virtual_bonds").

  Returns:
    residue_bonds: Dict that maps resname -> list of Bond tuples.
    residue_virtual_bonds: Dict that maps resname -> list of Bond tuples.
    residue_bond_angles: Dict that maps resname -> list of BondAngle tuples.
  """
  stereo_chemical_props_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), 'stereo_chemical_props.txt'
  )
  with open(stereo_chemical_props_path, 'rt') as f:
    stereo_chemical_props = f.read()
  lines_iter = iter(stereo_chemical_props.splitlines())
  # Load bond lengths.
  residue_bonds = {}
  next(lines_iter)  # Skip header line.
  for line in lines_iter:
    if line.strip() == '-':
      break
    bond, resname, length, stddev = line.split()
    atom1, atom2 = bond.split('-')
    if resname not in residue_bonds:
      residue_bonds[resname] = []
    residue_bonds[resname].append(
        Bond(atom1, atom2, float(length), float(stddev)))
  residue_bonds['UNK'] = []

  # Load bond angles.
  residue_bond_angles = {}
  next(lines_iter)  # Skip empty line.
  next(lines_iter)  # Skip header line.
  for line in lines_iter:
    if line.strip() == '-':
      break
    bond, resname, angle_degree, stddev_degree = line.split()
    atom1, atom2, atom3 = bond.split('-')
    if resname not in residue_bond_angles:
      residue_bond_angles[resname] = []
    residue_bond_angles[resname].append(
        BondAngle(atom1, atom2, atom3,
                  float(angle_degree) / 180. * np.pi,
                  float(stddev_degree) / 180. * np.pi))
  residue_bond_angles['UNK'] = []

  def make_bond_key(atom1_name, atom2_name):
    """Unique key to lookup bonds."""
    return '-'.join(sorted([atom1_name, atom2_name]))

  # Translate bond angles into distances ("virtual bonds").
  residue_virtual_bonds = {}
  for resname, bond_angles in residue_bond_angles.items():
    # Create a fast lookup dict for bond lengths.
    bond_cache = {}
    for b in residue_bonds[resname]:
      bond_cache[make_bond_key(b.atom1_name, b.atom2_name)] = b
    residue_virtual_bonds[resname] = []
    for ba in bond_angles:
      bond1 = bond_cache[make_bond_key(ba.atom1_name, ba.atom2_name)]
      bond2 = bond_cache[make_bond_key(ba.atom2_name, ba.atom3name)]

      # Compute distance between atom1 and atom3 using the law of cosines
      # c^2 = a^2 + b^2 - 2ab*cos(gamma).
      gamma = ba.angle_rad
      length = np.sqrt(bond1.length**2 + bond2.length**2
                       - 2 * bond1.length * bond2.length * np.cos(gamma))

      # Propagation of uncertainty assuming uncorrelated errors.
      dl_outer = 0.5 / length
      dl_dgamma = (2 * bond1.length * bond2.length * np.sin(gamma)) * dl_outer
      dl_db1 = (2 * bond1.length - 2 * bond2.length * np.cos(gamma)) * dl_outer
      dl_db2 = (2 * bond2.length - 2 * bond1.length * np.cos(gamma)) * dl_outer
      stddev = np.sqrt((dl_dgamma * ba.stddev)**2 +
                       (dl_db1 * bond1.stddev)**2 +
                       (dl_db2 * bond2.stddev)**2)
      residue_virtual_bonds[resname].append(
          Bond(ba.atom1_name, ba.atom3name, length, stddev))

  return (residue_bonds,
          residue_virtual_bonds,
          residue_bond_angles)


# Between-residue bond lengths for general bonds (first element) and for Proline
# (second element).
between_res_bond_length_c_n = [1.329, 1.341]
between_res_bond_length_stddev_c_n = [0.014, 0.016]

# Between-residue cos_angles.
between_res_cos_angles_c_n_ca = [-0.5203, 0.0353]  # degrees: 121.352 +- 2.315
between_res_cos_angles_ca_c_n = [-0.4473, 0.0311]  # degrees: 116.568 +- 1.995

# This mapping is used when we need to store atom data in a format that requires
# fixed atom data size for every residue (e.g. a numpy array).
atom_types = [
    'OP2', "C5'", "C2'", 'N2', 'N9', 'C2', "C3'", "O2'", 'C6', 'O2', 'O4', "C4'", 'N7', 'C4', "O4'", 'N6', 'OP1', 'N4', 'C8', 'N1', "O3'", "C1'", 'C5', 'O6', "O5'", 'N3', 'P'
]
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)  # := 37.

# A compact atom encoding with 14 columns
# pylint: disable=line-too-long
# pylint: disable=bad-whitespace
restype_name_to_atom14_names = {
  'A': ["P", "OP1", "OP2", "O5\'", "C5\'", "C4\'", "O4\'", "C1\'", "N9", "C4", "N3", "C2", "N1", "C6", "N6", "C5", "N7", "C8", "C2\'", "O2\'", "C3\'", "O3\'", ""],
  'C': ["P", "OP1", "OP2", "O5\'", "C5\'", "C4\'", "O4\'", "C1\'", "N1", "C6", "C2", "O2", "N3", "C4", "N4", "C5", "C2\'", "O2\'", "C3\'", "O3\'", "", "", ""],
  'G': ["P", "OP1", "OP2",  "O5\'", "C5\'", "C4\'", "O4\'", "C1\'", "N9", "C4", "N3", "C2", "N2", "N1", "C6", "O6", "C5", "N7", "C8", "C2\'", "O2\'", "C3\'", "O3\'"],
  'U': ["P", "OP1", "OP2", "O5\'", "C5\'", "C4\'", "O4\'", "C1\'", "N1", "C6", "C2", "O2", "N3", "C4", "O4", "C5", "C2\'", "O2\'", "C3\'", "O3\'", "", "", ""]
}
# pylint: enable=line-too-long
# pylint: enable=bad-whitespace


# This is the standard residue order when coding AA type as a number.
# Reproduce it by taking 3-letter AA codes and sorting them alphabetically.
# restypes = [
#     'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
#     'S', 'T', 'W', 'Y', 'V'
# ]

restypes = ['A', 'C', 'G', 'U']
restype_order = {restype: i for i, restype in enumerate(restypes)}
restype_num = len(restypes)  # := 20.
unk_restype_index = restype_num  # Catch-all index for unknown restypes.

restypes_with_x = restypes + ['X']
restype_order_with_x = {restype: i for i, restype in enumerate(restypes_with_x)}


def sequence_to_onehot(
    sequence: str,
    mapping: Mapping[str, int],
    map_unknown_to_x: bool = False) -> np.ndarray:
  """Maps the given sequence into a one-hot encoded matrix.

  Args:
    sequence: An amino acid sequence.
    mapping: A dictionary mapping amino acids to integers.
    map_unknown_to_x: If True, any amino acid that is not in the mapping will be
      mapped to the unknown amino acid 'X'. If the mapping doesn't contain
      amino acid 'X', an error will be thrown. If False, any amino acid not in
      the mapping will throw an error.

  Returns:
    A numpy array of shape (seq_len, num_unique_aas) with one-hot encoding of
    the sequence.

  Raises:
    ValueError: If the mapping doesn't contain values from 0 to
      num_unique_aas - 1 without any gaps.
  """
  num_entries = max(mapping.values()) + 1

  if sorted(set(mapping.values())) != list(range(num_entries)):
    raise ValueError('The mapping must have values from 0 to num_unique_aas-1 '
                     'without any gaps. Got: %s' % sorted(mapping.values()))

  one_hot_arr = np.zeros((len(sequence), num_entries), dtype=np.int32)

  for aa_index, aa_type in enumerate(sequence):
    if map_unknown_to_x:
      if aa_type.isalpha() and aa_type.isupper():
        aa_id = mapping.get(aa_type, mapping['X'])
      else:
        raise ValueError(f'Invalid character in the sequence: {aa_type}')
    else:
      aa_id = mapping[aa_type]
    one_hot_arr[aa_index, aa_id] = 1

  return one_hot_arr


restype_1to3 = {
    'A': 'A',
    'C': 'C',
    'G': 'G',
    'U': 'U'
}


# NB: restype_3to1 differs from Bio.PDB.protein_letters_3to1 by being a simple
# 1-to-1 mapping of 3 letter names to one letter names. The latter contains
# many more, and less common, three letter names as keys and maps many of these
# to the same one letter name (including 'X' and 'U' which we don't use here).
restype_3to1 = {v: k for k, v in restype_1to3.items()}

# Define a restype name for all unknown residues.
unk_restype = 'UNK'

# resnames = [[restype_1to3[r] for r in restypes] + [unk_restype]]
resnames = ['A', 'C', 'G', 'U']
resname_to_idx = {resname: i for i, resname in enumerate(resnames)}


# The mapping here uses hhblits convention, so that B is mapped to D, J and O
# are mapped to X, U is mapped to C, and Z is mapped to E. Other than that the
# remaining 20 amino acids are kept in alphabetical order.
# There are 2 non-amino acid codes, X (representing any amino acid) and
# "-" representing a missing amino acid in an alignment.  The id for these
# codes is put at the end (20 and 21) so that they can easily be ignored if
# desired.
HHBLITS_AA_TO_ID = {
    'A': 0,
    'B': 2,
    'C': 1,
    'D': 2,
    'E': 3,
    'F': 4,
    'G': 5,
    'H': 6,
    'I': 7,
    'J': 20,
    'K': 8,
    'L': 9,
    'M': 10,
    'N': 11,
    'O': 20,
    'P': 12,
    'Q': 13,
    'R': 14,
    'S': 15,
    'T': 16,
    'U': 1,
    'V': 17,
    'W': 18,
    'X': 20,
    'Y': 19,
    'Z': 3,
    '-': 21,
}

# Partial inversion of HHBLITS_AA_TO_ID.
ID_TO_HHBLITS_NT = {
    0: 'A',
    1: 'C',
    2: 'G',
    3: 'U'
}

restypes_with_x_and_gap = restypes
MAP_HHBLITS_AATYPE_TO_OUR_AATYPE = tuple(
    restypes_with_x_and_gap.index(ID_TO_HHBLITS_NT[i])
    for i in range(len(restypes_with_x_and_gap)))


def _make_standard_atom_mask() -> np.ndarray:
  """Returns [num_res_types, num_atom_types] mask array."""
  # +1 to account for unknown (all 0s).
  mask = np.zeros([restype_num + 1, atom_type_num], dtype=np.int32)
  for restype, restype_letter in enumerate(restypes):
    restype_name = restype_letter
    atom_names = residue_atoms[restype_name]
    for atom_name in atom_names:
      atom_type = atom_order[atom_name]
      mask[restype, atom_type] = 1
  return mask


STANDARD_ATOM_MASK = _make_standard_atom_mask()


# A one hot representation for the first and second atoms defining the axis
# of rotation for each chi-angle in each residue.
def chi_angle_atom(atom_index: int) -> np.ndarray:
  """Define chi-angle rigid groups via one-hot representations."""
  chi_angles_index = {}
  one_hots = []

  for k, v in chi_angles_atoms.items():
    indices = [atom_types.index(s[atom_index]) for s in v]
    indices.extend([-1]*(4-len(indices)))
    chi_angles_index[k] = indices

  for r in restypes:
    res3 = restype_1to3[r]
    one_hot = np.eye(atom_type_num)[chi_angles_index[res3]]
    one_hots.append(one_hot)

  one_hots.append(np.zeros([4, atom_type_num]))  # Add zeros for residue `X`.
  one_hot = np.stack(one_hots, axis=0)
  one_hot = np.transpose(one_hot, [0, 2, 1])

  return one_hot

chi_atom_1_one_hot = chi_angle_atom(1)
chi_atom_2_one_hot = chi_angle_atom(2)

# An array like chi_angles_atoms but using indices rather than names.
chi_angles_atom_indices = [chi_angles_atoms[restype_1to3[r]] for r in restypes]
chi_angles_atom_indices = tree.map_structure(
    lambda atom_name: atom_order[atom_name], chi_angles_atom_indices)
chi_angles_atom_indices = np.array([
    chi_atoms + ([[0, 0, 0, 0]] * (4 - len(chi_atoms)))
    for chi_atoms in chi_angles_atom_indices])

# Mapping from (res_name, atom_name) pairs to the atom's chi group index
# and atom index within that group.
chi_groups_for_atom = collections.defaultdict(list)
for res_name, chi_angle_atoms_for_res in chi_angles_atoms.items():
  for chi_group_i, chi_group in enumerate(chi_angle_atoms_for_res):
    for atom_i, atom in enumerate(chi_group):
      chi_groups_for_atom[(res_name, atom)].append((chi_group_i, atom_i))
chi_groups_for_atom = dict(chi_groups_for_atom)


def _make_rigid_transformation_4x4(ex, ey, translation):
  """Create a rigid 4x4 transformation matrix from two axes and transl."""
  # Normalize ex.
  ex_normalized = ex / np.linalg.norm(ex)

  # make ey perpendicular to ex
  ey_normalized = ey - np.dot(ey, ex_normalized) * ex_normalized
  ey_normalized /= np.linalg.norm(ey_normalized)

  # compute ez as cross product
  eznorm = np.cross(ex_normalized, ey_normalized)
  m = np.stack([ex_normalized, ey_normalized, eznorm, translation]).transpose()
  m = np.concatenate([m, [[0., 0., 0., 1.]]], axis=0)
  return m


# create an array with (restype, atomtype) --> rigid_group_idx
# and an array with (restype, atomtype, coord) for the atom positions
# and compute affine transformation matrices (4,4) from one rigid group to the
# previous group
restype_atom37_to_rigid_group = np.zeros([4, 37], dtype=np.int)
restype_atom37_mask = np.zeros([4, 37], dtype=np.float32)
restype_atom37_rigid_group_positions = np.zeros([4, 37, 3], dtype=np.float32)
restype_atom14_to_rigid_group = np.zeros([4, 23], dtype=np.int)
restype_atom14_mask = np.zeros([4, 23], dtype=np.float32)
restype_atom14_rigid_group_positions = np.zeros([4, 23, 3], dtype=np.float32)
restype_rigid_group_default_frame = np.zeros([4, 8, 4, 4], dtype=np.float32)


def _make_rigid_group_constants():
  """Fill the arrays above."""
  for restype, restype_letter in enumerate(restypes):
    resname = restype_1to3[restype_letter]
    for atomname, group_idx, atom_position in rigid_group_atom_positions[
        resname]:
      atomtype = atom_order[atomname]
      restype_atom37_to_rigid_group[restype, atomtype] = group_idx
      restype_atom37_mask[restype, atomtype] = 1
      restype_atom37_rigid_group_positions[restype, atomtype, :] = atom_position

      atom14idx = restype_name_to_atom14_names[resname].index(atomname)
      restype_atom14_to_rigid_group[restype, atom14idx] = group_idx
      restype_atom14_mask[restype, atom14idx] = 1
      restype_atom14_rigid_group_positions[restype,
                                           atom14idx, :] = atom_position

  for restype, restype_letter in enumerate(restypes):
    resname = restype_1to3[restype_letter]
    atom_positions = {name: np.array(pos) for name, _, pos
                      in rigid_group_atom_positions[resname]}

    # backbone to backbone is the identity transform
    restype_rigid_group_default_frame[restype, 0, :, :] = np.eye(4)

    # pre-omega-frame to backbone (currently dummy identity matrix)
    restype_rigid_group_default_frame[restype, 1, :, :] = np.eye(4)

    # phi-frame to backbone
    mat = _make_rigid_transformation_4x4(
        ex=atom_positions['P'] - atom_positions['C4\''],
        ey=np.array([1., 0., 0.]),
        translation=atom_positions['P'])
    restype_rigid_group_default_frame[restype, 2, :, :] = mat

    # psi-frame to backbone
    mat = _make_rigid_transformation_4x4(
        ex=atom_positions['C5\''] - atom_positions['C4\''],
        ey=atom_positions['C4\''] - atom_positions['P'],
        translation=atom_positions['C5\''])
    restype_rigid_group_default_frame[restype, 3, :, :] = mat

    # chi1-frame to backbone
    if chi_angles_mask[restype][0]:
      base_atom_names = chi_angles_atoms[resname][0]
      base_atom_positions = [atom_positions[name] for name in base_atom_names]
      mat = _make_rigid_transformation_4x4(
          ex=base_atom_positions[2] - base_atom_positions[1],
          ey=base_atom_positions[0] - base_atom_positions[1],
          translation=base_atom_positions[2])
      restype_rigid_group_default_frame[restype, 4, :, :] = mat

    # chi2-frame to chi1-frame
    # chi3-frame to chi2-frame
    # chi4-frame to chi3-frame
    # luckily all rotation axes for the next frame start at (0,0,0) of the
    # previous frame
    for chi_idx in range(0, 1):
      if chi_angles_mask[restype][chi_idx]:
        axis_end_atom_name = chi_angles_atoms[resname][chi_idx][2]
        axis_end_atom_position = atom_positions[axis_end_atom_name]
        mat = _make_rigid_transformation_4x4(
            ex=axis_end_atom_position,
            ey=np.array([-1., 0., 0.]),
            translation=axis_end_atom_position)
        restype_rigid_group_default_frame[restype, 4 + chi_idx, :, :] = mat


_make_rigid_group_constants()


def make_atom14_dists_bounds(overlap_tolerance=1.5,
                             bond_length_tolerance_factor=15):
  """compute upper and lower bounds for bonds to assess violations."""
  restype_atom14_bond_lower_bound = np.zeros([21, 14, 14], np.float32)
  restype_atom14_bond_upper_bound = np.zeros([21, 14, 14], np.float32)
  restype_atom14_bond_stddev = np.zeros([21, 14, 14], np.float32)
  residue_bonds, residue_virtual_bonds, _ = load_stereo_chemical_props()
  for restype, restype_letter in enumerate(restypes):
    resname = restype_1to3[restype_letter]
    atom_list = restype_name_to_atom14_names[resname]

    # create lower and upper bounds for clashes
    for atom1_idx, atom1_name in enumerate(atom_list):
      if not atom1_name:
        continue
      atom1_radius = van_der_waals_radius[atom1_name[0]]
      for atom2_idx, atom2_name in enumerate(atom_list):
        if (not atom2_name) or atom1_idx == atom2_idx:
          continue
        atom2_radius = van_der_waals_radius[atom2_name[0]]
        lower = atom1_radius + atom2_radius - overlap_tolerance
        upper = 1e10
        restype_atom14_bond_lower_bound[restype, atom1_idx, atom2_idx] = lower
        restype_atom14_bond_lower_bound[restype, atom2_idx, atom1_idx] = lower
        restype_atom14_bond_upper_bound[restype, atom1_idx, atom2_idx] = upper
        restype_atom14_bond_upper_bound[restype, atom2_idx, atom1_idx] = upper

    # overwrite lower and upper bounds for bonds and angles
    for b in residue_bonds[resname] + residue_virtual_bonds[resname]:
      atom1_idx = atom_list.index(b.atom1_name)
      atom2_idx = atom_list.index(b.atom2_name)
      lower = b.length - bond_length_tolerance_factor * b.stddev
      upper = b.length + bond_length_tolerance_factor * b.stddev
      restype_atom14_bond_lower_bound[restype, atom1_idx, atom2_idx] = lower
      restype_atom14_bond_lower_bound[restype, atom2_idx, atom1_idx] = lower
      restype_atom14_bond_upper_bound[restype, atom1_idx, atom2_idx] = upper
      restype_atom14_bond_upper_bound[restype, atom2_idx, atom1_idx] = upper
      restype_atom14_bond_stddev[restype, atom1_idx, atom2_idx] = b.stddev
      restype_atom14_bond_stddev[restype, atom2_idx, atom1_idx] = b.stddev
  return {'lower_bound': restype_atom14_bond_lower_bound,  # shape (21,14,14)
          'upper_bound': restype_atom14_bond_upper_bound,  # shape (21,14,14)
          'stddev': restype_atom14_bond_stddev,  # shape (21,14,14)
         }
