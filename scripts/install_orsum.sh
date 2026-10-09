#!/bin/bash
# Install orsum 1.8.0 (https://github.com/ozanozisik/orsum) into a virtual environment.
# orsum is three scripts, not a pip package; its dependencies (numpy, scipy, pandas,
# matplotlib, seaborn) come from requirements.txt. Scripts are verified by SHA-256 and
# copied verbatim except the shebang, which points at the environment's own Python so
# orsum runs there even when the environment is not activated.
#   usage: scripts/install_orsum.sh [venv]   (default: .venv)
set -euo pipefail

VENV="$(realpath -s "${1:-.venv}")"
VERSION=1.8.0
URL="https://github.com/ozanozisik/orsum/archive/refs/tags/${VERSION}.tar.gz"
declare -A SHA256=(
    [orsum.py]=7fb17bfe130d715f3a873a1c57e4a5128a4f8e0a82e07d288909184ed80e46fb
    [termCombinationLib.py]=7e5192df67629d1cd7916cec5bd690a3209695417e720bd55422af6ed9d5c39f
    [plotFunctions.py]=8004fc4f2cdec2f99dc80cf2acf977f5526fa5136f85b8b5c446ba08101b6592
)

if [[ ! -x "$VENV/bin/python" ]]; then
    echo "No Python environment at $VENV" >&2
    exit 1
fi
"$VENV/bin/python" -c "import numpy, scipy, pandas, matplotlib, seaborn" || {
    echo "orsum needs numpy, scipy, pandas, matplotlib and seaborn in $VENV" >&2
    exit 1
}

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fsSL --retry 3 "$URL" | tar xz -C "$tmp"
src="$tmp/orsum-${VERSION}"
for file in "${!SHA256[@]}"; do
    echo "${SHA256[$file]}  $src/$file" | sha256sum --check --quiet || {
        echo "Checksum mismatch for orsum $VERSION $file" >&2
        exit 1
    }
done

for file in termCombinationLib.py plotFunctions.py; do
    install -m 644 "$src/$file" "$VENV/bin/$file"
done
{ echo "#!$VENV/bin/python"; tail -n +2 "$src/orsum.py"; } > "$VENV/bin/orsum.py"
chmod 755 "$VENV/bin/orsum.py"
mkdir -p "$VENV/share/orsum-${VERSION}"
cp "$src/LICENSE" "$src/README.md" "$src/NEWS.md" "$VENV/share/orsum-${VERSION}/"
echo "orsum ${VERSION} from ${URL}; scripts verbatim except the orsum.py shebang." \
    > "$VENV/share/orsum-${VERSION}/INSTALL.txt"
echo "Installed orsum $("$VENV/bin/orsum.py" --version) into $VENV/bin"
