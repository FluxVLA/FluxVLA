# FluxVLA vendor patch: register this vendored package under the short alias
# ``_c3`` so legacy Cosmos3 imports inside compact vendored modules resolve.
import sys as _sys
import fluxvla.models.third_party_models.cosmos3 as _self  # noqa: F401

_sys.modules.setdefault('_c3', _self)
