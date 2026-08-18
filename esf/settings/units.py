"""The single shared pint unit registry for the esf package.

pint Quantities from different registries cannot be combined, and each
registry is expensive to build, so every module must use this one:

    from esf.settings.units import ureg, Q_
"""

from pint import UnitRegistry

ureg = UnitRegistry()
Q_ = ureg.Quantity
