#!/usr/bin/env python3
"""Compute transformed normalization statistics from LeRobot parquet data."""

from fluxvla.datasets.utils.transformed_statistics import (PROFILES, Profile,
                                                           compute_statistics,
                                                           main)

__all__ = ['PROFILES', 'Profile', 'compute_statistics', 'main']

if __name__ == '__main__':
    main()
