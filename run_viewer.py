"""Launch the CT/MR registration viewer.

    python run_viewer.py                       # empty window, load via buttons
    python run_viewer.py --folder sample_data  # auto-detect CT, MR and REG
    python run_viewer.py --ct CT_DIR --mr MR_DIR --reg REG.dcm
"""
import sys

from ctmr_reg.gui import main

if __name__ == "__main__":
    sys.exit(main())
