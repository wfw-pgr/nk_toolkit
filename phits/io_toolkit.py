import re
import pandas as pd
import pathlib


# ========================================================= #
# ===  read__phitsANGLE                                 === #
# ========================================================= #
def read__phitsANGEL( inpFile=None ) -> list[pd.DataFrame]:
    """Read numeric table blocks from a PHITS/ANGEL output file.
    This parser extracts numeric blocks that appear after an ANGEL table header. 

    Args:
        path: Path to the PHITS/ANGEL output file.

    Returns:
        A list of DataFrames. Each DataFrame corresponds to one numeric
        block found in the file.

    Raises:
        ValueError: If no numeric blocks are found.
    """
    NUM_RE         = re.compile( r"^[\s+-]?(?:\d|\.\d)" )
    path           = pathlib.Path( inpFile )
    blocks         = []
    pending_header = None
    rows           = []
    in_block       = False
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            stripped = line.strip()

            if stripped.startswith("h:"):
                pending_header = None
                rows = []
                in_block = False
                continue

            if stripped.startswith("#") and not in_block:
                candidate = stripped.lstrip("#").strip().split()

                if candidate:
                    pending_header = candidate
                continue

            if NUM_RE.match(stripped):
                values = stripped.split()

                try:
                    numeric_values = [float(value) for value in values]
                except ValueError:
                    continue

                if NUM_RE.match(stripped):
                    values = stripped.split()
                    
                try:
                    numeric_values = [float(value) for value in values]
                except ValueError:
                    continue

                if not in_block:
                    if pending_header is not None and len(pending_header) == len(numeric_values):
                        columns = pending_header
                    else:
                        columns = [
                            f"col{i + 1}"
                            for i in range(len(numeric_values))
                        ]

                    rows = []
                    in_block = True
                    pending_header = None
                    
                rows.append(numeric_values)
                continue
            
                if in_block:
                    rows.append(numeric_values)

                continue

            if in_block:
                blocks.append(pd.DataFrame(rows, columns=columns))
                rows = []
                pending_header = None
                in_block = False

    if in_block:
        blocks.append(pd.DataFrame(rows, columns=columns))

    if not blocks:
        raise ValueError("No numeric blocks were found in the file.")

    return(blocks)


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    inpFile = "out/fluence_n_energy.dat"
    ret     = read__phitsANGEL( inpFile=inpFile )
    
