#!/bin/sh
set -eu

target=${1:-"$PWD/sentinel-evc-install"}
profile=${SENTINEL_PROFILE:-all}
python=${PYTHON:-python3}
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
SENTINEL_PROFILE=$profile
PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export SENTINEL_PROFILE PYTHONPATH

if [ "$#" -eq 0 ] || [ "${1:-}" = "--interactive" ]; then
    exec "$python" -m sentinel_evc.install_wizard --source "$repo_root" --profile "$profile"
fi

exec "$python" -c 'from sentinel_evc.installation import install_system; import os,sys; r=install_system(sys.argv[1], profile=os.environ.get("SENTINEL_PROFILE", "all"), source=sys.argv[2], python_executable=os.environ.get("PYTHON", sys.executable), build_app=sys.platform=="darwin"); print(r.target)' "$target" "$repo_root"
