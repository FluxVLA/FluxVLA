# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


class FluxVLAExportError(Exception):
    """Base error for the public fluxvla-export API."""


class ContractError(FluxVLAExportError, ValueError):
    """A public export contract is incomplete or inconsistent."""
