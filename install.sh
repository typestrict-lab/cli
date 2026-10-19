#!/bin/sh
# Installs the typestrict CLI: curl -fsSL https://typestrict.ai/install.sh | sh
# Downloads the latest release for this system from github.com/typestrict-lab/cli, checks its SHA-256 and puts the binary
# in $TYPESTRICT_INSTALL_DIR (default ~/.local/bin).
set -eu

repo="typestrict-lab/cli"
dir="${TYPESTRICT_INSTALL_DIR:-$HOME/.local/bin}"

case "$(uname -s)" in
  Linux) os="unknown-linux-gnu" ;;
  Darwin) os="apple-darwin" ;;
  *) echo "typestrict: on Windows, download the zip from https://github.com/$repo/releases/latest" >&2; exit 1 ;;
esac
case "$(uname -m)" in
  x86_64 | amd64) arch="x86_64" ;;
  arm64 | aarch64) arch="aarch64" ;;
  *) echo "typestrict: no build for $(uname -m) yet" >&2; exit 1 ;;
esac

target="$arch-$os"
url="https://github.com/$repo/releases/latest/download/typestrict-$target.tar.gz"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "Downloading typestrict for $target..."
curl -fsSL "$url" -o "$tmp/typestrict.tar.gz"
curl -fsSL "$url.sha256" -o "$tmp/typestrict.tar.gz.sha256"
expected="$(cut -d ' ' -f 1 "$tmp/typestrict.tar.gz.sha256")"
if command -v sha256sum >/dev/null 2>&1; then actual="$(sha256sum "$tmp/typestrict.tar.gz" | cut -d ' ' -f 1)"
else actual="$(shasum -a 256 "$tmp/typestrict.tar.gz" | cut -d ' ' -f 1)"; fi
if [ "$expected" != "$actual" ]; then echo "typestrict: checksum mismatch, nothing installed" >&2; exit 1; fi

tar xzf "$tmp/typestrict.tar.gz" -C "$tmp"
mkdir -p "$dir"
mv "$tmp/typestrict" "$dir/typestrict"
chmod +x "$dir/typestrict"
echo "Installed $("$dir/typestrict" --version) to $dir/typestrict"
case ":$PATH:" in
  *":$dir:"*) ;;
  *) echo "Add it to your PATH: export PATH=\"$dir:\$PATH\"" ;;
esac
echo "Next: typestrict login"
