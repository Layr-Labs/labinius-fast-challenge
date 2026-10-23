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

fn doc_view(raw: &str) -> String {
    let mut view = String::new();
    for line in raw.split_inclusive('\n') {
        if let Some(rest) = line.strip_prefix("//!") { view.push_str("// "); view.push_str(rest); }
        else { view.push_str(line); }
    }
    view
}

fn source_view(raw: &str, root: &Path) -> String {
    let mut view = doc_view(raw);
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
        else { assert!(kind.is_file()); files.push(relative.to_owned()); }
    }
}

fn control_library_view(raw: &str, root: &Path, fold: &Path) -> (String, Vec<(String, PathBuf)>) {
    let mut view = doc_view(raw);
    let attribute = "#![allow(clippy::needless_range_loop)]";
    assert_eq!(view.matches(attribute).count(), 1);
    view = view.replace(attribute, "// control-entry owns #![allow(clippy::needless_range_loop)]");
    let names: Vec<_> = raw.lines().filter_map(|line| line.strip_prefix("pub mod ").and_then(|s| s.strip_suffix(';'))).collect();
    assert_eq!(names.len(), 20);
    let mut modules = Vec::new();
    for name in names {
        let path = if name == "fold" { fold.to_owned() } else {
            let file = root.join(format!("{name}.rs"));
            let directory = root.join(name).join("mod.rs");
            assert_ne!(file.is_file(), directory.is_file());
            if file.is_file() { file } else { directory }
        };
        let declaration = format!("pub mod {name};");
        assert_eq!(view.matches(&declaration).count(), 1);
        view = view.replace(&declaration, &format!("#[path = {:?}]\n{declaration}", path));
        modules.push((name.to_owned(), path));
    }
    (view, modules)
}

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest.parent().unwrap().join("proof_generate/src").canonicalize().unwrap();
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let main = root.join("main.rs");
    let library = manifest.parent().unwrap().join("labinius");
    let pcs = library.join("crates/pcs/src").canonicalize().unwrap();
    let crossfield = pcs.join("fields/crossfield.rs");
    let fold = pcs.join("fold.rs");
    let control_fold = manifest.join("src/control_fold_pr221.rs").canonicalize().unwrap();
    let control_entry = manifest.join("src/control_lib.rs");
    println!("cargo:rerun-if-changed={}", root.display());
    println!("cargo:rerun-if-changed={}", library.display());
    println!("cargo:rerun-if-changed={}", manifest.join("src").display());
    println!("cargo:rerun-if-env-changed=LABINIUS_DIAGNOSTIC_PAIR_KEY");
    assert_eq!(sha256(&main), "076fbb9674f0e721614c195d71b1e74789549fff809a53d49c8b57a5894f5fae");
    assert_eq!(sha256(&crossfield), "3abefc57bc2658545d40abe3666be0082cf7d6949b923c2884d1a7b302f9dacb");
    assert_eq!(sha256(&fold), "ad04c03eff53ddaa25ca58e24344c8b617a01e35f0f57097979b0331c5e96dab");
    assert_eq!(sha256(&control_fold), "ba57ba24491694f631e36c03dd3d410f11e4efc65ef3ed7f384516d51fe8d5ef");
    let mut files = Vec::new();
    library_sources(&library, &library, &mut files);
    files.sort();
    let mut candidate_manifest = String::new();
    let mut control_manifest = String::new();
    for relative in &files {
        let path = library.join(relative);
        candidate_manifest.push_str(&format!("{}\t{}\t{}\n", relative.display(), fs::metadata(&path).unwrap().len(), sha256(&path)));
        let path = if relative == Path::new("crates/pcs/src/fold.rs") { &control_fold } else { &path };
        control_manifest.push_str(&format!("{}\t{}\t{}\n", relative.display(), fs::metadata(path).unwrap().len(), sha256(path)));
    }
    let candidate_manifest_path = out.join("library-source-manifest.txt");
    let control_manifest_path = out.join("control-library-source-manifest.txt");
    fs::write(&candidate_manifest_path, candidate_manifest).unwrap();
    fs::write(&control_manifest_path, control_manifest).unwrap();
    let candidate_hash = sha256(&candidate_manifest_path);
    let control_hash = sha256(&control_manifest_path);
    assert_eq!(candidate_hash, "153858f4c53814daadcef85464655d7df56166fbe39a4e62b4323853e1d69af4");
    assert_eq!(control_hash, "e3a3554e79a1e1e8816f10608b10872fc08634c93522e200ba675d78e6dd64f5");
    let mut bindings = format!("main_sha256={} crossfield_sha256={} fold_sha256={} control_fold_sha256={}\nlibrary_source_files={} library_source_sha256={}\ncontrol_library_source_files={} control_library_source_sha256={}\nsource_root={} control_out_dir={}\nrustflags={:?} encoded_rustflags={:?}\n",
        sha256(&main), sha256(&crossfield), sha256(&fold), sha256(&control_fold), files.len(), candidate_hash,
        files.len(), control_hash, root.display(), out.display(), env::var("LABINIUS_DIAGNOSTIC_RUSTFLAGS"),
        env::var("LABINIUS_DIAGNOSTIC_ENCODED_RUSTFLAGS"));
    bindings.push_str(&format!("control_entry_sha256={} native_anchor=extern-crate-labinius-as-underscore\n", sha256(&control_entry)));
    bindings.push_str(&format!("native_build_source_sha256={} logq=48\n", sha256(&library.join("crates/pcs/build.rs"))));
    for name in ["keygen", "modulus_choice", "parameters", "rq"] {
        let path = root.join(format!("{name}.rs"));
        let raw = fs::read_to_string(&path).unwrap();
        assert!(!raw.contains("labinius_control::"));
        let destination = out.join(format!("{name}.rs"));
        fs::write(&destination, raw.replace("labinius::", "labinius_control::")).unwrap();
        bindings.push_str(&format!("private_source={name} raw_sha256={} control_view_sha256={}\n", sha256(&path), sha256(&destination)));
    }
    let library_root = pcs.join("lib.rs");
    let (control, modules) = control_library_view(&fs::read_to_string(&library_root).unwrap(), &pcs, &control_fold);
    let control_path = out.join("context-control-lib.rs");
    fs::write(&control_path, control).unwrap();
    bindings.push_str(&format!("control_library_root_raw_sha256={} control_library_root_view_sha256={}\n", sha256(&library_root), sha256(&control_path)));
    for (name, path) in modules {
        bindings.push_str(&format!("control_module={name} path={} sha256={}\n", path.display(), sha256(&path)));
    }
    let raw = fs::read_to_string(&main).unwrap();
    assert_eq!(raw.matches("const PREFIX_SETS: usize = 64;").count(), 1);
    assert_eq!(raw.matches("cf::LeanSwitch::msg(self)").count(), 1);
    assert_eq!(raw.matches("labinius::fold::fold_witness_bitsliced(").count(), 1);
    assert!(!raw.contains("labinius_control::"));
    let baseline = raw.replace("labinius::", "labinius_control::");
    for (name, selected, library_hash, fold_hash, source) in [
        ("context-candidate.rs", "labinius", &candidate_hash, sha256(&fold), source_view(&raw, &root)),
        ("context-baseline.rs", "labinius_control", &control_hash, sha256(&control_fold), source_view(&baseline, &out)),
    ] {
        let path = out.join(name);
        fs::write(&path, source).unwrap();
        bindings.push_str(&format!("source={name} sha256={}\n", sha256(&path)));
        bindings.push_str(&format!("library_view={name} selected_crate={selected} selected_api=fold_witness_bitsliced library_source_sha256={library_hash} fold_sha256={fold_hash}\n"));
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
