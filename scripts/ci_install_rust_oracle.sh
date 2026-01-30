#!/bin/bash
set -euxo pipefail

if [ -f /etc/oracle-release ]; then
    cat /etc/oracle-release
elif [ -f /etc/redhat-release ]; then
    cat /etc/redhat-release
fi

uname -m

if command -v sudo >/dev/null 2>&1; then
    sudo dnf -y install dnf-plugins-core
    sudo dnf repolist all
    sudo dnf config-manager --set-enabled ol9-codeready-builder-yum-remote
    sudo dnf -y install openssl-libs openssl-devel protobuf-compiler protobuf-devel jq bc gcc gcc-c++ make curl pkgconf-pkg-config perl
else
    dnf -y install dnf-plugins-core
    dnf repolist all
    dnf config-manager --set-enabled ol9-codeready-builder-yum-remote
    dnf -y install openssl-libs openssl-devel protobuf-compiler protobuf-devel jq bc gcc gcc-c++ make curl pkgconf-pkg-config perl
fi

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.90

source "$HOME/.cargo/env"

rustc --version
cargo --version
protoc --version
