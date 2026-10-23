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

fn library_sources(root: &Path, directory: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(directory).unwrap() {
        let path = entry.unwrap().path();
        let relative = path.strip_prefix(root).unwrap();
        if relative == Path::new("crates/pcs/labrador/libobj")
            || relative == Path::new("crates/pcs/labrador/liblabrador.a") { continue; }
        let kind = fs::symlink_metadata(&path).unwrap().file_type();
        assert!(!kind.is_symlink(), "source tree has no symlinks");
        if kind.is_dir() { library_sources(root, &path, files); }
        else { assert!(kind.is_file()); files.push(path.strip_prefix(root).unwrap().to_owned()); }
    }
}

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest.parent().unwrap().join("proof_generate/src").canonicalize().unwrap();
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let main = root.join("main.rs");
    let library = manifest.parent().unwrap().join("labinius");
    let crossfield = library.join("crates/pcs/src/fields/crossfield.rs");
    let fold = library.join("crates/pcs/src/fold.rs");
    println!("cargo:rerun-if-changed={}", root.display());
    println!("cargo:rerun-if-changed={}", library.display());
    println!("cargo:rerun-if-env-changed=LABINIUS_DIAGNOSTIC_PAIR_KEY");
    assert_eq!(sha256(&main), "076fbb9674f0e721614c195d71b1e74789549fff809a53d49c8b57a5894f5fae");
    assert_eq!(sha256(&crossfield), "3abefc57bc2658545d40abe3666be0082cf7d6949b923c2884d1a7b302f9dacb");
    assert_eq!(sha256(&fold), "ba57ba24491694f631e36c03dd3d410f11e4efc65ef3ed7f384516d51fe8d5ef");
    let mut files = Vec::new();
    library_sources(&library, &library, &mut files);
    files.sort();
    let mut library_manifest = String::new();
    for relative in &files {
        let path = library.join(relative);
        library_manifest.push_str(&format!("{}\t{}\t{}\n", relative.display(), fs::metadata(&path).unwrap().len(), sha256(&path)));
    }
    let library_manifest_path = out.join("library-source-manifest.txt");
    fs::write(&library_manifest_path, library_manifest).unwrap();
    let library_hash = sha256(&library_manifest_path);
    assert_eq!(library_hash, "e3a3554e79a1e1e8816f10608b10872fc08634c93522e200ba675d78e6dd64f5");
    let mut bindings = format!("main_sha256={} crossfield_sha256={} fold_sha256={}\nlibrary_source_files={} library_source_sha256={}\nrustflags={:?} encoded_rustflags={:?}\n",
        sha256(&main), sha256(&crossfield), sha256(&fold), files.len(), library_hash,
        env::var("LABINIUS_DIAGNOSTIC_RUSTFLAGS"), env::var("LABINIUS_DIAGNOSTIC_ENCODED_RUSTFLAGS"));
    let raw = fs::read_to_string(&main).unwrap();
    assert_eq!(raw.matches("const PREFIX_SETS: usize = 64;").count(), 1);
    assert_eq!(raw.matches("cf::LeanSwitch::msg(self)").count(), 1);
    let scored = "labinius::fold::fold_witness_bitsliced(";
    let control = "labinius::fold::fold_witness_bitsliced_baseline(";
    assert_eq!(raw.matches(scored).count(), 1);
    assert_eq!(raw.matches(control).count(), 0);
    let baseline = raw.replace(scored, control);
    for (name, api, source) in [
        ("context-candidate.rs", "fold_witness_bitsliced", raw),
        ("context-baseline.rs", "fold_witness_bitsliced_baseline", baseline),
    ] {
        let path = out.join(name);
        fs::write(&path, source_view(&source, &root)).unwrap();
        bindings.push_str(&format!("source={name} sha256={}\n", sha256(&path)));
        bindings.push_str(&format!("library_view={name} selected_api={api} library_source_sha256={library_hash} fold_sha256={}\n", sha256(&fold)));
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
