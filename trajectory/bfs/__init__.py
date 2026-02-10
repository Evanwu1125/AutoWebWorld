"""
BFS (Breadth-First Search) Traversal Module

This module provides tools for traversing FSMs using BFS algorithm
to generate all shortest paths and action sequences.

Main components:
- bfs.py: Core BFS algorithm for FSM traversal
- bfs_action.py: Action-level BFS with parameter generation
- normalize.py: FSM normalization utilities
- split_filters.py: Filter splitting for complex conditions
- filter_bfs_mapping.py: Filter and map BFS results
- general_params_provider.py: General parameter generation
- gui_mapping.py: GUI element mapping utilities
"""

from .bfs import Action, Page, BFSNode, deep_get, deep_set

__all__ = [
    'Action',
    'Page',
    'BFSNode',
    'deep_get',
    'deep_set',
]

__version__ = '1.0.0'

