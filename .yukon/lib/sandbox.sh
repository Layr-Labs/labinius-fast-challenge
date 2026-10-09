#!/usr/bin/env bash

if [[ -n "${LABINIUS_SANDBOX_SH_LOADED:-}" ]]; then
  return 0
fi
readonly LABINIUS_SANDBOX_SH_LOADED=1

require_host_command() {
  local path=$1
  if [[ ! -x "$path" ]]; then
    echo "required host tool is not executable: $path" >&2
    return 1
  fi
}

require_runner_temp_path() {
  local path=$1
  local parent resolved_parent resolved_temp
  [[ "$path" = /* ]] || {
    echo "state path must be absolute: $path" >&2
    return 1
  }
  resolved_temp=$(realpath "$RUNNER_TEMP")
  if [[ -L "$path" ]]; then
    echo "state path must not be a symlink: $path" >&2
    return 1
  fi
  case "$path" in
    "$RUNNER_TEMP"/*) ;;
    *)
      echo "state path must remain beneath RUNNER_TEMP: $path" >&2
      return 1
      ;;
  esac
  parent=$(dirname "$path")
  while [[ "$parent" != "$RUNNER_TEMP" && ! -e "$parent" ]]; do
    parent=$(dirname "$parent")
  done
  resolved_parent=$(realpath "$parent")
  if [[ "$resolved_parent" != "$resolved_temp" && "$resolved_parent" != "$resolved_temp"/* ]]; then
    echo "state path escapes RUNNER_TEMP through an existing parent: $path" >&2
    return 1
  fi
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

regular_file_size() {
  local path=$1
  if [[ ! -f "$path" || -L "$path" ]]; then
    echo "proof must be a regular non-symlink file" >&2
    return 1
  fi
  stat -c %s -- "$path"
}

system_library_binds() {
  local -n bind_args=$1
  local path
  for path in /lib /lib64; do
    if [[ -e "$path" ]]; then
      bind_args+=(--ro-bind "$path" "$path")
    fi
  done
  bind_args+=(--dir /etc)
  for path in /etc/ld.so.cache /etc/ld.so.conf; do
    if [[ -f "$path" ]]; then
      bind_args+=(--ro-bind "$path" "$path")
    fi
  done
}

network_config_binds() {
  local -n bind_args=$1
  local path
  bind_args+=(--dir /etc/ssl)
  if [[ -d /etc/ssl/certs ]]; then
    bind_args+=(--ro-bind /etc/ssl/certs /etc/ssl/certs)
  fi
  for path in /etc/resolv.conf /etc/hosts /etc/nsswitch.conf /etc/gai.conf; do
    if [[ -f "$path" ]]; then
      bind_args+=(--ro-bind "$path" "$path")
    fi
  done
}

run_contestant_build() {
  local solution_root=$1
  local target_dir=$2
  local cargo_home=$3
  local network_mode=$4
  shift 4

  local -a args=(
    --die-with-parent
    --new-session
    --cap-drop ALL
    --unshare-pid
    --unshare-ipc
    --unshare-uts
    --clearenv
    --proc /proc
    --dev /dev
    --tmpfs /tmp
    --dir /workspace
    --dir /workspace/solution
    --ro-bind "$solution_root" /workspace/solution
    --bind "$target_dir" /workspace/solution/proof_generate/target
    --dir /cargo-home
    --bind "$cargo_home" /cargo-home
    --dir /opt
    --dir /opt/cargo
    --ro-bind /opt/cargo/bin /opt/cargo/bin
    --ro-bind /opt/rustup /opt/rustup
    --ro-bind /bin /bin
    --dir /usr
    --ro-bind /usr/bin /usr/bin
    --ro-bind /usr/include /usr/include
    --ro-bind /usr/lib /usr/lib
    --ro-bind /usr/libexec /usr/libexec
    --setenv PATH /opt/cargo/bin:/usr/bin
    --setenv HOME /tmp
    --setenv RUSTUP_HOME /opt/rustup
    --setenv CARGO_HOME /cargo-home
    --setenv CARGO_TARGET_DIR /workspace/solution/proof_generate/target
    --setenv CC /usr/bin/gcc-13
    --setenv SSL_CERT_FILE /etc/ssl/certs/ca-certificates.crt
    --setenv CARGO_TERM_COLOR always
    --chdir /workspace/solution/proof_generate
  )
  system_library_binds args
  case "$network_mode" in
    fetch)
      network_config_binds args
      ;;
    offline)
      args+=(--unshare-net)
      ;;
    *)
      echo "unknown contestant network mode: $network_mode" >&2
      return 2
      ;;
  esac
  bwrap "${args[@]}" -- "$@"
}

run_contestant_prover() {
  local prover=$1
  local workdir=$2
  local seed=$3
  local proof=$4

  local -a args=(
    --die-with-parent
    --new-session
    --cap-drop ALL
    --unshare-pid
    --unshare-ipc
    --unshare-uts
    --unshare-net
    --clearenv
    --proc /proc
    --dev /dev
    --tmpfs /tmp
    --dir /app
    --ro-bind "$prover" /app/labinius-fast
    --dir /usr
    --ro-bind /usr/lib /usr/lib
    --dir /work
    --bind "$workdir" /work
    --setenv HOME /tmp
    --setenv TMPDIR /tmp
    --setenv LABINIUS_SEED "$seed"
    --chdir /work
  )
  system_library_binds args
  bwrap "${args[@]}" -- /app/labinius-fast --prove "$proof"
}
