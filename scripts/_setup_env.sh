#!/usr/bin/env bash
# Shared ROS 2 environment bootstrap. This file is sourced by run_*.sh.
# Packages must be built locally with colcon before using these scripts.

_ws_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
_ws_overlay="$_ws_root/install/local_setup.bash"

if [[ ! -f "$_ws_overlay" ]]; then
  printf 'ERROR: workspace is not built: %s\n' "$_ws_root" >&2
  printf 'Run: cd %s && colcon build --symlink-install\n' "$_ws_root" >&2
  exit 1
fi

# A workspace copied or moved after colcon build may have stale symlinks.
_ws_overlay_target="$(readlink -f -- "$_ws_overlay")"
case "$_ws_overlay_target" in
  "$_ws_root/install/"*) ;;
  *)
    printf 'ERROR: install overlay points outside the current workspace: %s\n' "$_ws_overlay_target" >&2
    printf 'Delete only the current workspace build/install/log and rebuild in place.\n' >&2
    exit 1
    ;;
esac

# Generated colcon setup files may access unset variables such as
# COLCON_CURRENT_PREFIX; never source them under bash nounset.
case "$-" in
  *u*) _ws_restore_nounset=true; set +u ;;
  *)   _ws_restore_nounset=false ;;
esac
source "$_ws_overlay"
if [[ "$_ws_restore_nounset" == true ]]; then set -u; fi
unset _ws_restore_nounset _ws_overlay_target

if ! command -v ros2 >/dev/null 2>&1; then
  printf 'ERROR: ros2 unavailable. Source your ROS 2 Jazzy environment first.\n' >&2
  exit 1
fi

_pkg_prefix="$(ros2 pkg prefix "$1" 2>/dev/null)" || {
  printf 'ERROR: ROS 2 package "%s" was not found.\n' "$1" >&2
  printf 'Rebuild: cd %s && colcon build --symlink-install\n' "$_ws_root" >&2
  exit 1
}

# Reject accidental loading of an identically named package from an older
# (possibly deleted) workspace such as ~/.local/share/Trash/files/...
case "$_pkg_prefix" in
  "$_ws_root/install"|"$_ws_root/install/"*) ;;
  *)
    printf 'ERROR: ROS 2 resolves package "%s" from a different workspace:\n  %s\n' "$1" "$_pkg_prefix" >&2
    printf 'Expected a package installed under: %s/install\n' "$_ws_root" >&2
    printf 'Use a clean terminal, remove stale ROS sources, and rebuild this workspace.\n' >&2
    exit 1
    ;;
esac
unset _pkg_prefix _ws_overlay _ws_root
