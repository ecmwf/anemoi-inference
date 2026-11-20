#!/usr/bin/env bash
# Local run of the "Test against PyPI" workflow (.github/workflows/python-pull-request.yml).
# Usage: tools/test-against-pypi.sh [python-version ...]   (default: 3.11 3.12 3.13)
set -euxo pipefail
cd "$(dirname "$0")/.."

versions=("$@")
[ ${#versions[@]} -eq 0 ] && versions=(3.11 3.12 3.13)

# Makes the tests behave as in CI, e.g. skip those needing S3 credentials.
export GITHUB_WORKFLOW=local
# files_for_tests resolves test data from here, as in CI.
export GITHUB_WORKSPACE=$PWD
unset PYTHONPATH

banner() {
    { set +x; } 2>/dev/null
    local rule; rule=$(printf '%.0s#' {1..72})
    printf '\n%s\n%s\n%s\n\n' "${rule}" "$*" "${rule}" >&2
    set -x
}

tmp=$(mktemp -d)
banner "Venvs go to ${tmp}
It is removed only if all checks pass; on failure it is kept for inspection."

# pre-commit fixes files in place, so it runs on a copy of the working tree, untracked files included.
# The -e test drops tracked files deleted in the working tree.
lint=${tmp}/lint; mkdir "${lint}"
{ set +x; } 2>/dev/null
git ls-files -z --cached --others --exclude-standard |
    while IFS= read -r -d '' f; do if [ -e "${f}" ]; then printf '%s\0' "${f}"; fi; done |
    tar --null -T - -cf - | tar -xf - -C "${lint}"
set -x
# In the fresh repo every file counts as added, so check-added-large-files would flag files already on main.
(cd "${lint}" && git init -q && git add -A && SKIP=no-commit-to-branch,check-added-large-files uvx pre-commit run --all-files --show-diff-on-failure)
rm -rf "${lint}"

# uv-managed Pythons on macOS trust only /etc/ssl/cert.pem, which lacks the issuer of the ECMWF object store.
run() {  # run <python-version> <extras> <pytest args...>
    local py=${1} extras=${2}; shift 2
    local venv=${tmp}/venv-${py}-${extras//,/-}
    echo "=== Python ${py} [${extras}]"
    uv venv --python "${py}" "${venv}"
    uv pip install --python "${venv}" ".[${extras}]"
    (export PATH="${venv}/bin:${PATH}" VIRTUAL_ENV="${venv}" SSL_CERT_FILE="$("${venv}/bin/python" -c "import certifi; print(certifi.where())")" && pytest -v "$@") || { echo "=== FAILED, venv kept in ${venv}" >&2; exit 1; }
    rm -rf "${venv}"
}

export UV_TORCH_BACKEND=auto
for py in "${versions[@]}"; do run "${py}" all,tests; done

rm -rf "${tmp}"
banner "All checks passed; removed ${tmp}"
