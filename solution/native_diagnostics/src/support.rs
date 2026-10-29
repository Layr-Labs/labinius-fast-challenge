#![allow(dead_code)]
use std::{fs, path::Path, process::Command};

pub const SCHEMA: &str = "candidate-b-native-v1";
pub const ROLES: [&str; 2] = ["baseline-control", "candidate-b"];
pub const TABLE_BYTES: usize = 1 << 28;
pub const PROOF_BYTES: usize = 13_958_612;
pub const MODULUS: u128 = 40_772_829_184;

pub fn sha256(path: &Path) -> String {
    let mut command = if cfg!(target_os = "macos") {
        let mut c = Command::new("/usr/bin/shasum");
        c.args(["-a", "256"]);
        c
    } else {
        Command::new("/usr/bin/sha256sum")
    };
    let output = command.arg(path).output().expect("hash bound artifact");
    assert!(output.status.success());
    let text = String::from_utf8(output.stdout).unwrap();
    let hash = text.split_whitespace().next().unwrap();
    assert!(hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit()));
    hash.to_owned()
}

pub fn quote(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < ' ' => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

pub fn files(root: &Path, dir: &Path, found: &mut Vec<std::path::PathBuf>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        let rel = path.strip_prefix(root).unwrap();
        let name = path.file_name().unwrap().to_str().unwrap();
        if matches!(
            name,
            "target" | "libobj" | "liblabrador.a" | ".git" | "__pycache__"
        ) {
            continue;
        }
        let kind = fs::symlink_metadata(&path).unwrap().file_type();
        assert!(!kind.is_symlink(), "source cannot contain symbolic links");
        if kind.is_dir() {
            files(root, &path, found);
        } else {
            assert!(kind.is_file());
            found.push(rel.to_owned());
        }
    }
}

pub fn copy_native(from: &Path, to: &Path) {
    fs::create_dir(to).expect("fresh private native source directory");
    for entry in fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name();
        if name == "libobj" || name == "liblabrador.a" {
            continue;
        }
        assert!(!entry.file_type().unwrap().is_symlink());
        if entry.file_type().unwrap().is_dir() {
            copy_native(&entry.path(), &to.join(name));
        } else {
            fs::copy(entry.path(), to.join(name)).unwrap();
        }
    }
}
