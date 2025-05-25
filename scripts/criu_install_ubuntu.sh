#!/bin/bash
set -e
if [[ $EUID -ne 0 ]]; then
  echo "This script must be run as root (use sudo)" >&2
  exit 1
fi

# See https://criu.org/Installation#Dependencies
apt install -y build-essential 
apt install -y libprotobuf-dev libprotobuf-c-dev protobuf-c-compiler protobuf-compiler python3-protobuf
apt install -y pkg-config
apt install -y uuid-dev
apt install -y libbsd-dev
apt install -y iproute2
apt install -y libnftables-dev
apt install -y libcap-dev
apt install -y libnl-3-dev libnet-dev
apt install -y libgnutls28-dev
apt install -y libaio-dev
apt install -y libdrm-dev

CRIU_VERSION=4.1
wget "https://github.com/checkpoint-restore/criu/archive/refs/tags/v${CRIU_VERSION}.tar.gz"
tar -zxf "v${CRIU_VERSION}.tar.gz"
cd "criu-${CRIU_VERSION}/"
make

apt install -y asciidoc xmlto 
make install
ldconfig

python -m pip install google
python -m pip install "protobuf<4"  # avoid compatibility issues
python -m pip install "criu-${CRIU_VERSION}/lib"  # install pycriu
python -m pip install "criu-${CRIU_VERSION}/crit"
