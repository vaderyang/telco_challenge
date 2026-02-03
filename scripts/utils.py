#!/usr/bin/env python3
"""
Auto-install missing dependencies helper.

Usage at the top of scripts:
    from utils import ensure_deps
    ensure_deps(['aiohttp', 'requests', 'pandas'])
"""

import subprocess
import sys


def ensure_deps(packages: list, quiet: bool = True):
    """
    Check if packages are installed, install missing ones automatically.
    
    Args:
        packages: List of package names to check/install
        quiet: If True, suppress pip output
    
    Example:
        ensure_deps(['aiohttp', 'requests', 'pandas>=1.5.0'])
    """
    missing = []
    
    for pkg in packages:
        # Handle version specifiers
        pkg_name = pkg.split('>=')[0].split('==')[0].split('<')[0].strip()
        # Handle package name mapping (import name != pip name)
        import_name = {
            'scikit-learn': 'sklearn',
            'pillow': 'PIL',
            'pyyaml': 'yaml',
        }.get(pkg_name, pkg_name)
        
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"[auto-install] Installing missing packages: {', '.join(missing)}")
        cmd = [sys.executable, '-m', 'pip', 'install'] + missing
        if quiet:
            cmd.append('-q')
        try:
            subprocess.check_call(cmd)
            print(f"[auto-install] Successfully installed: {', '.join(missing)}")
        except subprocess.CalledProcessError as e:
            print(f"[auto-install] Failed to install: {e}")
            sys.exit(1)


# Quick check function for common packages
def ensure_common():
    """Ensure common packages for this project are installed."""
    ensure_deps([
        'requests',
        'aiohttp', 
        'pandas',
        'numpy',
    ])


if __name__ == '__main__':
    # Test
    ensure_deps(['requests', 'aiohttp'])
    print("All dependencies OK!")
