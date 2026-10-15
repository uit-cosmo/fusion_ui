"""What the app may not import: anything of fusion_scripts (L14).

Its modules are top-level packages (fusion_scripts has none of its own), so an import of one of them
is an import from it. ``imaging_methods.decorrelation`` is a submodule of imaging_methods, a different
thing from the top-level ``decorrelation``, and is allowed; so is ``fusion_ui.config``, a different
thing from the top-level ``config``. Imports are read from the syntax tree, so a string or a comment
that says one is not one.
"""

import ast

#: The top-level packages and modules of fusion_scripts that the app and its tests may not import.
FUSION_SCRIPTS = frozenset(
    {
        "fusion_scripts",
        "density_scan",
        "decorrelation",
        "twodca_manuscript",
        "plotting_scripts",
        "waveform_analysis",
        "individual_events",
        "figure_scripts",
        "phantom_events",
        "tools",
        "config",
    }
)


def imported_names(source):
    """``[(level, dotted name)]`` of every import in ``source``: ``from a.b import c`` is ``a.b`` and ``a.b.c``."""
    names = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names += [(0, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names.append((node.level, module))
            names += [
                (node.level, f"{module}.{alias.name}" if module else alias.name)
                for alias in node.names
            ]
    return names


def reaches_fusion_scripts(source):
    """The imports of ``source`` that go to fusion_scripts: a top-level package of it (an absolute import
    whose first part is one), or the app's retired ``core.fusion_scripts``, however it is spelled."""
    found = []
    for level, name in imported_names(source):
        parts = name.split(".")
        if level == 0 and parts[0] in FUSION_SCRIPTS:
            found.append(name)
        elif "fusion_scripts" in parts:
            found.append(name)
    return found
