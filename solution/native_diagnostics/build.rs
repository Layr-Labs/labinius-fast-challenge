use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
};
#[path = "src/support.rs"]
mod support;
use support::{files, sha256};

const CHANGED: [(&str, &str, &str); 3] = [
    (
        "simd/ntt/gen_small.rs",
        "ce04e601a6b3d721993e37dee26eac131d94c1b5c7e162d02f651fc89f675810",
        "40508a583f9209678041aa61db468f8b0acba4eb102a9ed00085122f0c74bba8",
    ),
    (
        "simd/sms_commit.rs",
        "a42894363e9285075edbd9ce663f300d43a9e229e8408cd2ce90ca152e329647",
        "e0c6e36b72ce651bb78d7678b82fd03855b80db375391b8ff771f642a1ccd593",
    ),
    (
        "simd/sms_slot_gen.rs",
        "994d4dec78a2ca264d07f01227c5bd26cfa8c75135540c4652c3d5fecd6f3f7a",
        "1541be110ec0ba192531aa3977477f802af910aa86b372658019bd41231d3ce2",
    ),
];

fn doc_view(raw: &str) -> String {
    raw.lines()
        .filter(|s| !s.starts_with("#![allow(clippy::needless_range_loop)]"))
        .map(|s| {
            if let Some(s) = s.strip_prefix("//!") {
                format!("// {s}\n")
            } else {
                format!("{s}\n")
            }
        })
        .collect()
}

// Keep source modules in place; redirect only the approved control branches.
fn module_view(raw: &str, root: &Path, overrides: &[(&str, PathBuf)]) -> String {
    let mut lines: Vec<String> = doc_view(raw).lines().map(str::to_owned).collect();
    let mut out = String::new();
    for i in 0..lines.len() {
        let Some(name) = lines[i]
            .strip_prefix("pub mod ")
            .and_then(|s| s.strip_suffix(';'))
        else {
            if !lines[i].starts_with("#[path=") && !lines[i].starts_with("#[path =") {
                out.push_str(&lines[i]);
                out.push('\n');
            }
            continue;
        };
        let path = if let Some((_, path)) = overrides.iter().find(|(n, _)| *n == name) {
            path.clone()
        } else if i > 0 && lines[i - 1].starts_with("#[path") {
            root.join(lines[i - 1].split('"').nth(1).expect("path attribute"))
        } else if root.join(format!("{name}.rs")).is_file() {
            root.join(format!("{name}.rs"))
        } else {
            root.join(name).join("mod.rs")
        };
        assert!(path.is_file(), "module source {}", path.display());
        out.push_str(&format!("#[path = {:?}]\npub mod {name};\n", path));
    }
    lines.clear();
    out
}

fn source_main(raw: &str, private: &Path) -> String {
    let mut view = doc_view(raw);
    for name in ["keygen", "modulus_choice", "parameters", "rq"] {
        let declaration = format!("mod {name};");
        assert_eq!(view.matches(&declaration).count(), 1);
        view = view.replace(
            &declaration,
            &format!(
                "#[path = {:?}]\n{declaration}",
                private.join(format!("{name}.rs"))
            ),
        );
    }
    view
}

fn main() {
    let helper = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap())
        .canonicalize()
        .unwrap();
    let solution = helper.parent().unwrap();
    let pcs = solution.join("labinius/crates/pcs/src");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    println!("cargo:rerun-if-changed={}", solution.display());
    for var in [
        "LABINIUS_DIAGNOSTIC_KEYDIR",
        "RUSTFLAGS",
        "CARGO_ENCODED_RUSTFLAGS",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    println!("cargo:rustc-link-arg-bins=-Wl,-z,max-page-size=0x200000");
    let mut binding = String::new();
    let mut bind = |kind: &str, name: &str, path: &Path| {
        assert!(fs::symlink_metadata(path).unwrap().file_type().is_file());
        binding.push_str(&format!(
            "{kind}\t{name}\t{}\t{}\t{}\n",
            sha256(path),
            fs::metadata(path).unwrap().len(),
            path.display()
        ));
    };
    let guard = helper.join("frozen-source.tsv");
    let expected = fs::read_to_string(&guard).expect("source must be frozen before build");
    let mut current = Vec::new();
    files(solution, solution, &mut current);
    current.sort();
    current.retain(|p| p != Path::new("native_diagnostics/frozen-source.tsv"));
    let rows: Vec<_> = expected.lines().collect();
    assert_eq!(
        rows.len(),
        current.len(),
        "source file set changed after freeze"
    );
    for (row, relative) in rows.iter().zip(&current) {
        let fields: Vec<_> = row.split('\t').collect();
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0], relative.to_str().unwrap());
        let path = solution.join(relative);
        assert_eq!(fields[1], fs::metadata(&path).unwrap().len().to_string());
        assert_eq!(
            fields[2],
            sha256(&path),
            "source changed after freeze: {}",
            relative.display()
        );
    }
    bind("source", "frozen-solution", &guard);
    let mut overrides = BTreeMap::new();
    for (relative, candidate, control) in CHANGED {
        let path = helper
            .join("control")
            .join(Path::new(relative).file_name().unwrap());
        assert_eq!(sha256(&pcs.join(relative)), candidate);
        assert_eq!(sha256(&path), control);
        overrides.insert(PathBuf::from(format!("crates/pcs/src/{relative}")), path);
    }
    let library = solution.join("labinius");
    let mut paths = Vec::new();
    files(&library, &library, &mut paths);
    paths.sort();
    for role in ["candidate", "control"] {
        let text: String = paths
            .iter()
            .map(|rel| {
                let path = if role == "control" {
                    overrides
                        .get(rel)
                        .cloned()
                        .unwrap_or_else(|| library.join(rel))
                } else {
                    library.join(rel)
                };
                format!(
                    "{}\t{}\t{}\n",
                    rel.display(),
                    fs::metadata(&path).unwrap().len(),
                    sha256(&path)
                )
            })
            .collect();
        let path = out.join(format!("{role}-library.tsv"));
        fs::write(&path, text).unwrap();
        bind("source", &format!("{role}-library"), &path);
    }
    let fixture = env::var_os("CARGO_FEATURE_FIXTURE").is_some();
    for (role, kernel, sms, slot) in [
        (
            "control",
            helper.join("control/gen_small.rs"),
            helper.join("control/sms_commit.rs"),
            helper.join("control/sms_slot_gen.rs"),
        ),
        (
            "fixture",
            pcs.join("simd/ntt/gen_small.rs"),
            pcs.join("simd/sms_commit.rs"),
            pcs.join("simd/sms_slot_gen.rs"),
        ),
    ] {
        let mut raw = doc_view(&fs::read_to_string(&sms).unwrap());
        let needle = "include!(\"sms_slot_gen.rs\");";
        assert_eq!(raw.matches(needle).count(), 1);
        raw = raw.replace(needle, &format!("include!({:?});", slot));
        if fixture {
            for (needle, insertion) in [
                (
                    "let chunk = eh * cph + (local - 1) / CHUNK_PAIRS;",
                    "diagnostic_bias(ws.bias[0]);",
                ),
                (
                    "if half == 1 || b + LOCK == range.end {",
                    "diagnostic_pending(&ws.pend, half);",
                ),
                (
                    "gen_small::intt_gen_batch32::<P>(pd);",
                    "diagnostic_inverse(pd, d, both);",
                ),
            ] {
                assert_eq!(raw.matches(needle).count(), 1);
                raw = if needle.starts_with("if half") {
                    raw.replace(needle, &format!("{insertion}\n{needle}"))
                } else {
                    raw.replace(needle, &format!("{needle}\n{insertion}"))
                };
            }
            raw.push_str(&fs::read_to_string(helper.join("src/sms_private_view.rs")).unwrap());
        }
        let sms_view = out.join(format!("{role}-sms.rs"));
        fs::write(&sms_view, raw).unwrap();
        bind("view", &format!("{role}-sms-raw"), &sms);
        bind("view", &format!("{role}-sms-slot"), &slot);
        bind("view", &format!("{role}-sms"), &sms_view);
        let ntt = out.join(format!("{role}-ntt.rs"));
        fs::write(
            &ntt,
            module_view(
                &fs::read_to_string(pcs.join("simd/ntt/mod.rs")).unwrap(),
                &pcs.join("simd/ntt"),
                &[("gen_small", kernel)],
            ),
        )
        .unwrap();
        let commit = out.join(format!("{role}-commit.rs"));
        fs::write(
            &commit,
            module_view(
                &fs::read_to_string(pcs.join("simd/commit.rs")).unwrap(),
                &pcs.join("simd"),
                &[("sms", sms_view)],
            ),
        )
        .unwrap();
        let simd = out.join(format!("{role}-simd.rs"));
        fs::write(
            &simd,
            module_view(
                &fs::read_to_string(pcs.join("simd/mod.rs")).unwrap(),
                &pcs.join("simd"),
                &[("ntt", ntt.clone()), ("commit", commit.clone())],
            ),
        )
        .unwrap();
        let lib = out.join(format!("{role}-lib.rs"));
        fs::write(
            &lib,
            module_view(
                &fs::read_to_string(pcs.join("lib.rs")).unwrap(),
                &pcs,
                &[("simd", simd.clone())],
            ),
        )
        .unwrap();
        for (name, path) in [
            ("ntt", ntt),
            ("commit", commit),
            ("simd", simd),
            ("lib", lib),
        ] {
            bind("view", &format!("{role}-{name}"), &path);
        }
    }
    let private = solution.join("proof_generate/src");
    for name in ["keygen", "modulus_choice", "parameters", "rq"] {
        let path = private.join(format!("{name}.rs"));
        let raw = fs::read_to_string(&path).unwrap();
        let target = out.join(format!("{name}.rs"));
        fs::write(&target, raw.replace("labinius::", "labinius_control::")).unwrap();
        bind("view", &format!("control-{name}"), &target);
    }
    let raw = fs::read_to_string(private.join("main.rs")).unwrap();
    assert_eq!(
        sha256(&private.join("main.rs")),
        "29bc502b630feba9d471356edc42d8feb3d1b9b06ad6453c90ee4261e97aa3e3"
    );
    for (name, source) in [
        ("candidate-a", source_main(&raw, &private)),
        (
            "best-control",
            source_main(&raw.replace("labinius::", "labinius_control::"), &out),
        ),
    ] {
        let path = out.join(format!("{name}.rs"));
        fs::write(&path, source).unwrap();
        bind("view", name, &path);
    }
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux")
        && env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("x86_64")
    {
        let keys = PathBuf::from(
            env::var_os("LABINIUS_DIAGNOSTIC_KEYDIR")
                .expect("actual parent-generated key directory"),
        );
        for (name, bytes) in [
            ("key.bin", 0),
            ("tail-key.bin", 0),
            ("hybrid-key.bin", 0),
            ("pair-key.bin", 0),
            ("sms-pairs.bin", 42_467_328),
            ("sms-mur.bin", 145_152),
            ("sms-mu.bin", 2592),
        ] {
            let source = keys.join(name);
            assert_eq!(fs::metadata(&source).unwrap().len(), bytes);
            let target = out.join(name);
            fs::copy(&source, &target).unwrap();
            assert_eq!(sha256(&source), sha256(&target));
            bind("key-parent", name, &source);
            bind("key", name, &target);
        }
    }
    fs::write(out.join("bindings.tsv"), binding).unwrap();
}
