#!/usr/bin/env bash

set -xeuf -o pipefail

source environment


sudo apt-get update
sudo apt-get install -y build-essential curl git lzip python3 python3-dev python3-pip python3-venv pipx

pipx install black
