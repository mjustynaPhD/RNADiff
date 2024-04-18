# import mmcif parser and load the cif structure
from Bio.PDB.MMCIFParser import MMCIFParser

parser = MMCIFParser(QUIET=True)
structure = parser.get_structure("example", "../bgsu_cifs/7RQB_1_1A_IL_7RQB_001.cif")
# iterate over residues. If the residue is A then print all atoms in the residue and break
all_atoms = []
for model in structure:
    for chain in model:
        for residue in chain:
            if residue.get_resname() == "U":
                atoms = [(atom.name, atom.coord) for atom in residue if atom.element != 'H']
                # atoms = ", ".join(atoms).replace("'", "\\'")
                atoms = dict(atoms)
                c4p = atoms['C4\'']
                # for at, coord in atoms.items():
                    # print(f'[\"{at}\", 0, {tuple(coord - c4p)}],')
                break
                # print(atoms)