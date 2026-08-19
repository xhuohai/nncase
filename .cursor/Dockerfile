# Base image for the nncase Cloud Agent development environment.
#
# Provides the stable system toolchain required to build nncase from source:
#   - GCC/G++ 14 (the C++ toolchain the conan profile pins, toolchains/x86_64-linux.profile.jinja)
#   - Ninja + CMake (native build system used by conan-release preset)
#   - Python 3.10 (matches CI, best wheel compatibility for the pinned test deps)
#   - .NET 7 SDK (Directory.Packages.props / csproj target net7.0)
#
# Repository-specific dependency installation (conan, dotnet restore/publish,
# pip) happens in .cursor/install.sh so it can see the checked-out source.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    DOTNET_CLI_TELEMETRY_OPTOUT=1 \
    DOTNET_NOLOGO=1 \
    DOTNET_ROOT=/usr/share/dotnet

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg git build-essential \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc-14 g++-14 \
        ninja-build cmake \
        python3.10 python3.10-venv python3.10-dev \
    && rm -rf /var/lib/apt/lists/*

# .NET 7 SDK (installed via the official dotnet-install script; not packaged for 24.04).
RUN curl -fsSL https://dot.net/v1/dotnet-install.sh -o /tmp/dotnet-install.sh \
    && chmod +x /tmp/dotnet-install.sh \
    && /tmp/dotnet-install.sh --channel 7.0 --install-dir "${DOTNET_ROOT}" \
    && ln -sf "${DOTNET_ROOT}/dotnet" /usr/local/bin/dotnet \
    && rm /tmp/dotnet-install.sh

# The ubuntu:24.04 image ships a non-root "ubuntu" user (uid 1000); ensure it exists.
RUN id ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash ubuntu

USER ubuntu
