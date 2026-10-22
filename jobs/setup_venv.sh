# Create or update .venv in the repository root; sourced by the cluster jobs.
# Installs with uv when available (uv.lock), otherwise python3 -m venv + pip, then orsum.
# Skips the install when pyproject.toml, uv.lock and requirements.txt are unchanged since
# the last setup. A lock file stops parallel jobs from installing at the same time.
#   PYTHON  interpreter for the pip fallback (default python3, needs 3.11 or newer)

setup_venv() {
    local stamp=.venv/.setup-stamp want
    want=$(cat pyproject.toml uv.lock requirements.txt scripts/install_orsum.sh 2>/dev/null | sha256sum | cut -d' ' -f1)
    exec 9> .venv.lock
    flock 9
    if [[ -x .venv/bin/python && -x .venv/bin/orsum.py && "$(cat "$stamp" 2>/dev/null)" == "$want" ]]; then
        echo "Using existing .venv ($(.venv/bin/python --version))"
    else
        echo "Setting up .venv"
        local uv
        uv=$(command -v uv || echo "$HOME/.local/bin/uv")
        if [[ -x "$uv" && -f uv.lock ]]; then
            "$uv" sync --locked --extra test --extra plots
        else
            local py=${PYTHON:-python3}
            "$py" -c 'import sys; sys.exit(sys.version_info < (3, 11))' \
                || { echo "$py is older than Python 3.11; set PYTHON=..." >&2; return 1; }
            [[ -x .venv/bin/python ]] || "$py" -m venv .venv
            .venv/bin/python -m pip install --upgrade pip
            .venv/bin/python -m pip install -r requirements.txt
        fi
        bash scripts/install_orsum.sh .venv
        echo "$want" > "$stamp"
    fi
    flock -u 9
    exec 9>&-
}

setup_venv
