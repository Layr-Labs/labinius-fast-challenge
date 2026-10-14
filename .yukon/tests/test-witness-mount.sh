#!/usr/bin/env bash
# The witness table mount is readable and not writable inside the prover sandbox.
set -euo pipefail
umask 077

script_dir=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=.yukon/lib/sandbox.sh
source "$script_dir/lib/sandbox.sh"

: "${RUNNER_TEMP:?RUNNER_TEMP is required}"
command -v bwrap >/dev/null
compiler=${CC:-cc}
command -v "$compiler" >/dev/null

root=$(mktemp -d "$RUNNER_TEMP/labinius-mount-test.XXXXXX")
cleanup() { rm -rf "$root"; }
trap cleanup EXIT

cat > "$root/ro-prover.c" <<'EOF'
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
    unsigned char buf[4];
    int fd;
    FILE *out;
    if (argc != 5 || strcmp(argv[1], "--prove") != 0 || strcmp(argv[3], "--table") != 0)
        return 10;
    if (strcmp(argv[4], "/input/witness-table.bin") != 0)
        return 11;
    fd = open(argv[4], O_RDONLY);
    if (fd < 0 || read(fd, buf, 4) != 4)
        return 2;
    close(fd);
    if (open(argv[4], O_WRONLY) >= 0)
        return 4;
    if (unlink(argv[4]) == 0)
        return 5;
    if (open("/input/other", O_CREAT | O_WRONLY, 0644) >= 0)
        return 6;
    out = fopen(argv[2], "wb");
    if (!out || fwrite(buf, 1, 4, out) != 4)
        return 7;
    fclose(out);
    return 0;
}
EOF
"$compiler" -O2 -o "$root/ro-prover" "$root/ro-prover.c"

mkdir -p "$root/table" "$root/work"
printf 'ABCD witness' > "$root/table/witness-table.bin"
chmod 444 "$root/table/witness-table.bin"
before=$(cat "$root/table/witness-table.bin")

run_contestant_prover \
  "$root/ro-prover" \
  "$root/work" \
  0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  /work/proof.bin \
  "$root/table/witness-table.bin"

[[ "$(cat "$root/table/witness-table.bin")" == "$before" ]]
[[ "$(cat "$root/work/proof.bin")" == "ABCD" ]]
echo "witness table is read-only inside the prover sandbox"
