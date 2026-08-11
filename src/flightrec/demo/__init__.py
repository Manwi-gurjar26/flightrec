"""A deliberately fallible demo agent, used for every measurement in this project.

You cannot build a debugger without something broken to debug. This package is
a three-tool research agent whose failures are *seeded and reproducible*, so
the same seed always produces the same wrong answer. That is what makes replay
fidelity and diff localization measurable rather than anecdotal.

It runs against a deterministic stub model by default, so anyone can reproduce
the numbers in the README without an API key.
"""
