"""Foundation layer: application settings and the read-only database gateway.

Everything else in the prototype builds on these two modules:

- ``config`` — typed settings loaded from ``.env`` (credentials kept secret).
- ``db``     — the single read-only PostgreSQL access point (parameterised,
               statement-timed, and guarded against non-read statements).
"""
