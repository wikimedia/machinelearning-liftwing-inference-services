#!/bin/bash
set -ex

# Provide a hipconfig shim so AITER can import. (see P94313#384150)
# The ROCm 7.2.0 install in this image ships without hipconfig, which
# AITER 0.1.13 requires at import time (get_hip_version runs `hipconfig --version`).
mkdir -p ~/bin
cat > ~/bin/hipconfig <<'EOF'
#!/bin/sh
echo "7.2.0"
EOF
chmod +x ~/bin/hipconfig
export PATH=~/bin:$PATH

# This file contains common environment variables that represents good
# defaults for Python libraries that model servers import.
source common_settings.sh
MODEL_SERVER_PATH="${1:-model_server/model.py}"
# Run the model server
exec /srv/venv/bin/python ${MODEL_SERVER_PATH} "${@:2}"
