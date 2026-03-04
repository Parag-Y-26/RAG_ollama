"""
Configuration package for NotebookLM.

Provides centralized application settings loaded from ``st.secrets``
and environment variables.  Import the singleton ``settings`` object
from ``config.settings`` wherever you need configuration values.

Modules
-------
settings
    Immutable ``Settings`` dataclass with every tuneable parameter.
"""
