use std::{env, fs, path::{Path, PathBuf}, process::Command};

fn sha256(path: &Path) -> String {
    let mut command = if cfg!(target_os = "macos") {
        let mut c = Command::new("shasum");
        c.args(["-a", "256"]);
        c
    } else { Command::new("sha256sum") };
    let output = command.arg(path).output().expect("hash source or public key");
    assert!(output.status.success(), "hash command failed");
    String::from_utf8(output.stdout).unwrap().split_whitespace().next().unwrap().to_owned()
}

fn source_view(raw: &str, root: &Path) -> String {
    let mut view = String::new();
    for line in raw.split_inclusive('\n') {
        if let Some(rest) = line.strip_prefix("//!") { view.push_str("// "); view.push_str(rest); }
        else { view.push_str(line); }
    }
    for name in ["keygen", "modulus_choice", "parameters", "rq"] {
        let declaration = format!("mod {name};");
        assert_eq!(view.matches(&declaration).count(), 1);
        view = view.replace(&declaration, &format!("#[path = {:?}]\n{declaration}", root.join(format!("{name}.rs"))));
    }
    view
}

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest.parent().unwrap().join("proof_generate/src").canonicalize().unwrap();
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let main = root.join("main.rs");
    let crossfield = manifest.parent().unwrap().join("labinius/crates/pcs/src/fields/crossfield.rs");
    println!("cargo:rerun-if-changed={}", root.display());
    println!("cargo:rerun-if-changed={}", crossfield.display());
    println!("cargo:rerun-if-env-changed=LABINIUS_DIAGNOSTIC_PAIR_KEY");
    assert_eq!(sha256(&main), "076fbb9674f0e721614c195d71b1e74789549fff809a53d49c8b57a5894f5fae");
    assert_eq!(sha256(&crossfield), "3abefc57bc2658545d40abe3666be0082cf7d6949b923c2884d1a7b302f9dacb");
    let mut bindings = format!("main_sha256={} crossfield_sha256={}\nrustflags={:?} encoded_rustflags={:?}\n",
        sha256(&main), sha256(&crossfield), env::var("LABINIUS_DIAGNOSTIC_RUSTFLAGS"),
        env::var("LABINIUS_DIAGNOSTIC_ENCODED_RUSTFLAGS"));
    let raw = fs::read_to_string(&main).unwrap();
    assert_eq!(raw.matches("const PREFIX_SETS: usize = 64;").count(), 1);
    let restored = raw.replace("const PREFIX_SETS: usize = 64;", "const PREFIX_SETS: usize = 32;");
    let restored_path = out.join("pr147-restored-main.rs");
    fs::write(&restored_path, &restored).unwrap();
    assert_eq!(sha256(&restored_path), "f9f152af10266b04d545a6c8012459236df760786b11386bbe856317a110cf19");
    assert_eq!(restored.matches("cf::LeanSwitch::msg(self)").count(), 1);
    let baseline = restored.replace("cf::LeanSwitch::msg(self)", "cf::LeanSwitch::msg_baseline(self)");
    for (name, source) in [("context-candidate.rs", raw), ("context-baseline.rs", baseline)] {
        let path = out.join(name);
        fs::write(&path, source_view(&source, &root)).unwrap();
        bindings.push_str(&format!("source={name} sha256={}\n", sha256(&path)));
    }
    if env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("x86_64")
        && env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        let pair = PathBuf::from(env::var_os("LABINIUS_DIAGNOSTIC_PAIR_KEY").expect("parent-generated public key path"));
        for name in ["key.bin", "tail-key.bin", "hybrid-key.bin", "pair-key.bin"] {
            let source = pair.parent().unwrap().join(name);
            assert!(fs::symlink_metadata(&source).unwrap().file_type().is_file());
            let bytes = fs::metadata(&source).unwrap().len();
            assert_eq!(bytes, if name == "pair-key.bin" { 21_233_664 } else { 0 });
            let destination = out.join(name);
            fs::copy(&source, &destination).unwrap();
            let hash = sha256(&source);
            assert_eq!(hash, sha256(&destination));
            bindings.push_str(&format!("key={name} bytes={bytes} sha256={hash}\n"));
            println!("cargo:rerun-if-changed={}", source.display());
        }
    }
    fs::write(out.join("context-bindings.txt"), bindings).unwrap();
}
