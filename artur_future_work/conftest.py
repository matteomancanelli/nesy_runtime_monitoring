"""Make `import src` resolve to THIS folder's src/ package.

While artur_future_work/ still lives inside the parent repo, the parent's
editable install also provides a package named `src`. Prepending this
directory guarantees the local copy wins, so `pytest` run from this folder
tests this folder's code. Once extracted into its own repo (with its own
`pip install -e .` in a fresh env), this shim is harmless but no longer
strictly needed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
