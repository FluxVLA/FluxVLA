#!/bin/bash
# Wrapper script to fix libffi conflict
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
exec python "$@"
