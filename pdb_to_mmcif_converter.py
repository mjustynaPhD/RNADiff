# a script to convert pdb files to mmcif files (format)
import Bio.PDB
import os
from tqdm import tqdm

def pdb_to_mmcif_converter(pdb_file, mmcif_file, entry_id):
    parser = Bio.PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(entry_id, pdb_file)
    io = Bio.PDB.MMCIFIO()
    io.set_structure(structure)
    io.save(mmcif_file)

if __name__ == '__main__':
    pdbs_path = "../bgsu_pdbs/"
    mmcif_path = "../bgsu_cifs/"
    pdb_files = os.listdir(pdbs_path)
    for pdb_file in tqdm(pdb_files):
        pdb_file = os.path.join(pdbs_path, pdb_file)
        mmcif_file = os.path.join(mmcif_path, pdb_file.split('/')[-1].split('.')[0] + '.cif')
        pdb_to_mmcif_converter(pdb_file, mmcif_file, entry_id=pdb_file.split('_')[0])
