"""
Global pytest config, fixtures, and helpers go here!
"""

# Standard
import os
import sys

# Make sure tests can import torchchat
sys.path.append(
    os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
)
