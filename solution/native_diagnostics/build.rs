use std::{env, fs, path::{Path, PathBuf}};
#[path = "src/support.rs"]
mod support;
use support::{files, sha256};

const MAIN: &str = "6f075cfc32687f6c227137dfc242edeb1412c18b8776e1d907bf5d62b7b93b12";
const OLD_MAIN: &str = "29bc502b630feba9d471356edc42d8feb3d1b9b06ad6453c90ee4261e97aa3e3";
const SMS: &str = "f0a5bc8c57bf08bad57f761a5a5b1e6ecf13443fd2c2693081578a76444accc8";
const OLD_SMS: &str = "e0c6e36b72ce651bb78d7678b82fd03855b80db375391b8ff771f642a1ccd593";

fn doc_view(raw: &str) -> String {
    raw.lines().filter(|s| !s.starts_with("#![allow(clippy::needless_range_loop)]"))
        .map(|s| if let Some(s) = s.strip_prefix("//!") { format!("// {s}\n") } else { format!("{s}\n") }).collect()
}
fn module_view(raw: &str, root: &Path, overrides: &[(&str, PathBuf)]) -> String {
    let lines: Vec<_> = doc_view(raw).lines().map(str::to_owned).collect();
    let mut out = String::new();
    for (i, line) in lines.iter().enumerate() {
        let Some(name) = line.strip_prefix("pub mod ").and_then(|s| s.strip_suffix(';')) else {
            if !line.starts_with("#[path=") && !line.starts_with("#[path =") { out.push_str(line); out.push('\n'); }
            continue;
        };
        let path = if let Some((_, p)) = overrides.iter().find(|(n, _)| *n == name) { p.clone() }
            else if i > 0 && lines[i-1].starts_with("#[path") { root.join(lines[i-1].split('"').nth(1).unwrap()) }
            else if root.join(format!("{name}.rs")).is_file() { root.join(format!("{name}.rs")) }
            else { root.join(name).join("mod.rs") };
        assert!(path.is_file(), "module source {}", path.display());
        out.push_str(&format!("#[path = {:?}]\npub mod {name};\n", path));
    }
    out
}
fn source_main(raw: &str, private: &Path) -> String {
    let mut view = doc_view(raw);
    for name in ["keygen", "modulus_choice", "parameters", "rq"] {
        let declaration = format!("mod {name};");
        assert_eq!(view.matches(&declaration).count(), 1);
        view = view.replace(&declaration, &format!("#[path = {:?}]\n{declaration}", private.join(format!("{name}.rs"))));
    }
    view
}
fn block<'a>(raw: &'a str, anchor: &str) -> &'a str {
    assert_eq!(raw.matches(anchor).count(), 1);
    let start = raw.find(anchor).unwrap();
    let open = start + raw[start..].find('{').unwrap();
    let mut depth = 0;
    for (at, byte) in raw.as_bytes().iter().enumerate().skip(open) {
        if *byte == b'{' { depth += 1; }
        if *byte == b'}' { depth -= 1; if depth == 0 { return &raw[start..=at]; } }
    }
    panic!("unclosed source block")
}
fn main() {
    let helper = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).canonicalize().unwrap();
    let solution = helper.parent().unwrap();
    let pcs = solution.join("labinius/crates/pcs/src");
    let private = solution.join("proof_generate/src");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    println!("cargo:rerun-if-changed={}", solution.display());
    for var in ["LABINIUS_DIAGNOSTIC_KEYDIR", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS"] { println!("cargo:rerun-if-env-changed={var}"); }
    println!("cargo:rustc-link-arg-bins=-Wl,-z,max-page-size=0x200000");
    let mut binding = String::new();
    let mut bind = |kind: &str, name: &str, path: &Path| {
        assert!(fs::symlink_metadata(path).unwrap().file_type().is_file());
        binding.push_str(&format!("{kind}\t{name}\t{}\t{}\t{}\n", sha256(path), fs::metadata(path).unwrap().len(), path.display()));
    };
    let guard = helper.join("frozen-source.tsv");
    let expected = fs::read_to_string(&guard).expect("source must be frozen before build");
    let mut current = Vec::new(); files(solution, solution, &mut current); current.sort();
    current.retain(|p| p != Path::new("native_diagnostics/frozen-source.tsv"));
    let rows: Vec<_> = expected.lines().collect(); assert_eq!(rows.len(), current.len(), "source set changed after freeze");
    for (row, rel) in rows.iter().zip(&current) {
        let f: Vec<_> = row.split('\t').collect(); assert_eq!(f.len(), 3); assert_eq!(f[0], rel.to_str().unwrap());
        let path = solution.join(rel); assert_eq!(f[1], fs::metadata(&path).unwrap().len().to_string()); assert_eq!(f[2], sha256(&path), "source changed: {}", rel.display());
    }
    bind("source", "frozen-solution", &guard);
    assert_eq!(sha256(&private.join("main.rs")), MAIN); assert_eq!(sha256(&helper.join("control/main.rs")), OLD_MAIN);
    assert_eq!(sha256(&pcs.join("simd/sms_commit.rs")), SMS); assert_eq!(sha256(&helper.join("control/sms_commit.rs")), OLD_SMS);
    let library = solution.join("labinius");
    let mut paths = Vec::new(); files(&library, &library, &mut paths); paths.sort(); assert_eq!(paths.len(), 199);
    for role in ["candidate", "control"] {
        let text: String = paths.iter().map(|rel| {
            let path = if role == "control" && rel == Path::new("crates/pcs/src/simd/sms_commit.rs") { helper.join("control/sms_commit.rs") } else { library.join(rel) };
            format!("{}\t{}\t{}\n", rel.display(), fs::metadata(&path).unwrap().len(), sha256(&path))
        }).collect();
        let path = out.join(format!("{role}-library.tsv")); fs::write(&path, text).unwrap(); bind("source", &format!("{role}-library"), &path);
        let mut text = String::new();
        for name in ["main", "keygen", "modulus_choice", "parameters", "rq"] {
            let path = if role == "control" && name == "main" { helper.join("control/main.rs") } else { private.join(format!("{name}.rs")) };
            text.push_str(&format!("proof_generate/src/{name}.rs\t{}\t{}\n", fs::metadata(&path).unwrap().len(), sha256(&path)));
        }
        let path = out.join(format!("{role}-main.tsv")); fs::write(&path, text).unwrap(); bind("source", &format!("{role}-main"), &path);
    }
    let fixture = env::var_os("CARGO_FEATURE_FIXTURE").is_some();
    for (role, sms, bridge) in [
        ("control", helper.join("control/sms_commit.rs"), "sms_control_view.rs"),
        ("fixture", pcs.join("simd/sms_commit.rs"), "sms_canonical_view.rs"),
    ] {
        let slot = pcs.join("simd/sms_slot_gen.rs");
        let mut raw = doc_view(&fs::read_to_string(&sms).unwrap());
        let needle = "include!(\"sms_slot_gen.rs\");"; assert_eq!(raw.matches(needle).count(), 1);
        raw = raw.replace(needle, &format!("include!({:?});", slot));
        if fixture { raw.push_str(&fs::read_to_string(helper.join("src").join(bridge)).unwrap()); }
        let sms_view = out.join(format!("{role}-sms.rs")); fs::write(&sms_view, raw).unwrap();
        bind("view", &format!("{role}-sms-raw"), &sms); bind("view", &format!("{role}-sms-slot"), &slot); bind("view", &format!("{role}-sms"), &sms_view);
        let commit = out.join(format!("{role}-commit.rs"));
        fs::write(&commit, module_view(&fs::read_to_string(pcs.join("simd/commit.rs")).unwrap(), &pcs.join("simd"), &[("sms", sms_view)])).unwrap();
        let simd = out.join(format!("{role}-simd.rs"));
        fs::write(&simd, module_view(&fs::read_to_string(pcs.join("simd/mod.rs")).unwrap(), &pcs.join("simd"), &[("commit", commit.clone())])).unwrap();
        let lib = out.join(format!("{role}-lib.rs"));
        fs::write(&lib, module_view(&fs::read_to_string(pcs.join("lib.rs")).unwrap(), &pcs, &[("simd", simd.clone())])).unwrap();
        for (name, path) in [("commit", commit), ("simd", simd), ("lib", lib)] { bind("view", &format!("{role}-{name}"), &path); }
    }
    for name in ["keygen", "modulus_choice", "parameters", "rq"] {
        let raw = fs::read_to_string(private.join(format!("{name}.rs"))).unwrap();
        let target = out.join(format!("{name}.rs")); fs::write(&target, raw.replace("labinius::", "labinius_control::")).unwrap(); bind("view", &format!("control-{name}"), &target);
    }
    for (name, path, control) in [("candidate-c", private.join("main.rs"), false), ("baseline-control", helper.join("control/main.rs"), true)] {
        let raw = fs::read_to_string(&path).unwrap(); bind("view", &format!("{name}-main-raw"), &path);
        let generated = if control { source_main(&raw.replace("labinius::", "labinius_control::"), &out) } else { source_main(&raw, &private) };
        let path = out.join(format!("{name}.rs")); fs::write(&path, generated).unwrap(); bind("view", name, &path);
    }
    if fixture {
        let raw = fs::read_to_string(private.join("main.rs")).unwrap();
        let aligned = "#[repr(C, align(64))]\nstruct Aligned64<T>(T);";
        let normalized = raw.replace("\r\n", "\n"); assert_eq!(normalized.matches(aligned).count(), 1);
        let mut view = format!("{aligned}\n{}\n", block(&normalized, "const fn sms_canonical_means"));
        for prefix in ["const SMS_MUR_LEN:", "static SMS_MUR:"] {
            let lines: Vec<_> = normalized.lines().filter(|l| l.starts_with(prefix)).collect(); assert_eq!(lines.len(), 1); view.push_str(lines[0]); view.push('\n');
        }
        view.push_str(&fs::read_to_string(helper.join("src/mean_static_view.rs")).unwrap());
        let path = out.join("mean-static.rs"); fs::write(&path, view).unwrap(); bind("view", "mean-static", &path);
    }
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") && env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("x86_64") {
        let keys = PathBuf::from(env::var_os("LABINIUS_DIAGNOSTIC_KEYDIR").expect("actual parent-generated keys"));
        for (name, bytes) in [("key.bin",0),("tail-key.bin",0),("hybrid-key.bin",0),("pair-key.bin",0),("sms-pairs.bin",42_467_328),("sms-mur.bin",145_152),("sms-mu.bin",2592)] {
            let source = keys.join(name); let target = out.join(name); let hash = sha256(&source);
            assert_eq!(fs::metadata(&source).unwrap().len(), bytes); fs::copy(&source, &target).unwrap();
            assert_eq!(sha256(&source), hash); assert_eq!(sha256(&target), hash); assert_eq!(fs::metadata(&target).unwrap().len(), bytes);
            bind("key-parent", name, &source); bind("key", name, &target);
        }
    }
    fs::write(out.join("bindings.tsv"), binding).unwrap();
}
