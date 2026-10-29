use crate::support::*;
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::Instant,
};

struct Record {
    seq: usize,
    file: fs::File,
}
impl Record {
    fn emit(&mut self, event: &str, fields: &[(&str, String)]) {
        self.seq += 1;
        let mut line = format!(
            "{{\"schema\":{},\"seq\":{},\"event\":{}",
            quote(SCHEMA),
            self.seq,
            quote(event)
        );
        for (name, value) in fields {
            line.push_str(&format!(",{}:{value}", quote(name)));
        }
        line.push('}');
        writeln!(self.file, "{line}").unwrap();
        println!("candidate-c-json {line}");
    }
    fn artifact(&mut self, namespace: &str, name: &str, path: &Path) {
        let path = path.canonicalize().unwrap();
        assert!(fs::symlink_metadata(&path).unwrap().file_type().is_file());
        self.emit(
            "artifact",
            &[
                ("namespace", quote(namespace)),
                ("name", quote(name)),
                ("path", quote(path.to_str().unwrap())),
                ("bytes", fs::metadata(&path).unwrap().len().to_string()),
                ("sha256", quote(&sha256(&path))),
            ],
        );
    }
}

fn bindings(record: &mut Record, namespace: &str, out: &Path) {
    let path = out.join("bindings.tsv");
    record.artifact(namespace, "bindings", &path);
    for row in fs::read_to_string(path).unwrap().lines() {
        let f: Vec<_> = row.split('\t').collect();
        assert_eq!(f.len(), 5);
        let path = Path::new(f[4]);
        assert_eq!(sha256(path), f[2]);
        assert_eq!(fs::metadata(path).unwrap().len().to_string(), f[3]);
        record.emit(
            "binding",
            &[
                ("namespace", quote(namespace)),
                ("kind", quote(f[0])),
                ("name", quote(f[1])),
                ("sha256", quote(f[2])),
                ("bytes", f[3].to_owned()),
                ("path", quote(f[4])),
            ],
        );
    }
}

fn native_link(record: &mut Record, namespace: &str, release: &Path, native: &Path) {
    let mut outputs = Vec::new();
    for entry in fs::read_dir(release.join("build")).unwrap() {
        let path = entry.unwrap().path().join("output");
        if path.is_file() {
            let text = fs::read_to_string(&path).unwrap();
            if text
                .lines()
                .any(|l| l == "cargo:rustc-link-lib=static=labinius_c")
            {
                outputs.push((path, text));
            }
        }
    }
    assert_eq!(outputs.len(), 1, "one actual native build");
    let (path, text) = outputs.pop().unwrap();
    let searches: Vec<_> = text
        .lines()
        .filter_map(|l| l.strip_prefix("cargo:rustc-link-search=native="))
        .map(Path::new)
        .collect();
    let libraries: Vec<_> = text
        .lines()
        .filter(|l| l.starts_with("cargo:rustc-link-lib="))
        .collect();
    assert_eq!(
        libraries,
        [
            "cargo:rustc-link-lib=static=labinius_c",
            "cargo:rustc-link-lib=static=labrador",
            "cargo:rustc-link-lib=m"
        ]
    );
    assert_eq!(searches.len(), 2);
    let c = path.parent().unwrap().join("out").canonicalize().unwrap();
    assert_eq!(searches[0].canonicalize().unwrap(), c);
    assert_eq!(
        searches[1].canonicalize().unwrap(),
        native.canonicalize().unwrap()
    );
    let stamp = native.join("libobj/.labinius_logq_stamp");
    assert_eq!(fs::read_to_string(&stamp).unwrap().trim(), "48");
    record.artifact(namespace, "native-link-metadata", &path);
    record.artifact(namespace, "native-c-archive", &c.join("liblabinius_c.a"));
    record.artifact(
        namespace,
        "native-labrador-archive",
        &native.join("liblabrador.a"),
    );
    record.artifact(namespace, "native-logq48-stamp", &stamp);
    let mut objects = Vec::new();
    let labrador_objects = native.join("libobj");
    for dir in [&c, &labrador_objects] {
        for e in fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.extension().is_some_and(|e| e == "o") {
                objects.push(p);
            }
        }
    }
    objects.sort();
    assert!(!objects.is_empty());
    for (i, path) in objects.iter().enumerate() {
        record.artifact(namespace, &format!("native-object-{i}"), path);
    }
}

// Bind the outer production/key-generator C builds as well as the three private targets.
fn production_native(record: &mut Record, outer: &Path) {
    let release = outer
        .ancestors()
        .nth(3)
        .expect("parent Cargo release directory");
    assert_eq!(release.file_name().unwrap(), "release");
    let mut outputs = Vec::new();
    for e in fs::read_dir(release.join("build")).unwrap() {
        let path = e.unwrap().path().join("output");
        if path.is_file() {
            let text = fs::read_to_string(&path).unwrap();
            if text
                .lines()
                .any(|s| s == "cargo:rustc-link-lib=static=labinius_c")
            {
                outputs.push((path, text));
            }
        }
    }
    outputs.sort_by(|a, b| a.0.cmp(&b.0));
    assert!(!outputs.is_empty());
    for (i, (metadata, text)) in outputs.iter().enumerate() {
        let dirs: Vec<_> = text
            .lines()
            .filter_map(|s| s.strip_prefix("cargo:rustc-link-search=native="))
            .map(Path::new)
            .collect();
        assert_eq!(dirs.len(), 2);
        let libraries: Vec<_> = text
            .lines()
            .filter(|s| s.starts_with("cargo:rustc-link-lib="))
            .collect();
        assert_eq!(
            libraries,
            [
                "cargo:rustc-link-lib=static=labinius_c",
                "cargo:rustc-link-lib=static=labrador",
                "cargo:rustc-link-lib=m"
            ]
        );
        assert_eq!(
            dirs[0].canonicalize().unwrap(),
            metadata
                .parent()
                .unwrap()
                .join("out")
                .canonicalize()
                .unwrap()
        );
        let stamp = dirs[1].join("libobj/.labinius_logq_stamp");
        assert_eq!(fs::read_to_string(&stamp).unwrap(), "48");
        for (name, path) in [
            ("metadata", metadata.clone()),
            ("c-archive", dirs[0].join("liblabinius_c.a")),
            ("labrador-archive", dirs[1].join("liblabrador.a")),
            ("logq48", stamp),
        ] {
            record.artifact("production-build", &format!("native-{i}-{name}"), &path);
        }
        let mut objects = Vec::new();
        for dir in [dirs[0].to_owned(), dirs[1].join("libobj")] {
            for e in fs::read_dir(dir).unwrap() {
                let path = e.unwrap().path();
                if path.extension().is_some_and(|s| s == "o") {
                    objects.push(path);
                }
            }
        }
        objects.sort();
        assert!(!objects.is_empty());
        for (j, path) in objects.iter().enumerate() {
            record.artifact("production-build", &format!("native-{i}-object-{j}"), path);
        }
    }
}

fn helper_out(release: &Path) -> PathBuf {
    let mut found = Vec::new();
    for e in fs::read_dir(release.join("build")).unwrap() {
        let out = e.unwrap().path().join("out");
        if out.join("bindings.tsv").is_file() {
            found.push(out);
        }
    }
    assert_eq!(found.len(), 1);
    found.pop().unwrap()
}

fn correctness(record: &mut Record, outer: &Path, work: &Path) {
    let helper = Path::new(env!("CARGO_MANIFEST_DIR"));
    let cargo = std::env::var("LABINIUS_DIAGNOSTIC_CARGO").unwrap();
    let host = std::env::var("LABINIUS_DIAGNOSTIC_HOST").unwrap();
    for (case, target, test, feature) in [
        (
            "focused",
            "candidate-c-focused",
            "candidate_c_checks::canonical_mean_equivalence",
            true,
        ),
        (
            "round-trip",
            "prover-round-trip",
            "tests::round_trip_and_reject_wrong_instance",
            false,
        ),
    ] {
        let target_dir = outer.join(format!("candidate-c-{case}-target"));
        assert!(!target_dir.exists());
        let native = outer.join(format!("candidate-c-{case}-labrador"));
        copy_native(&helper.join("../labinius/crates/pcs/labrador"), &native);
        let manifest = helper.join("Cargo.toml");
        let mut args = vec![
            "test",
            "--locked",
            "--offline",
            "--release",
            "--jobs",
            "1",
            "--target",
            &host,
            "-vv",
            "--manifest-path",
            manifest.to_str().unwrap(),
        ]
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
        if feature {
            args.extend(["--features".into(), "fixture".into()]);
        }
        args.extend([
            "--test".into(),
            target.into(),
            test.into(),
            "--".into(),
            "--exact".into(),
            "--nocapture".into(),
            "--test-threads=1".into(),
        ]);
        let command_path = work.join(format!("{case}.command"));
        fs::write(&command_path,format!("program={cargo:?}\nargs={args:?}\ntarget={target_dir:?}\nnative={native:?}\nkeydir={outer:?}\nRUSTFLAGS={:?}\nCARGO_ENCODED_RUSTFLAGS={:?}\nmanual_OUT_DIR=false\n",std::env::var("RUSTFLAGS"),std::env::var("CARGO_ENCODED_RUSTFLAGS"))).unwrap();
        record.artifact(case, "command", &command_path);
        let output = Command::new(&cargo)
            .args(&args)
            .current_dir(helper)
            .env("CARGO_TARGET_DIR", &target_dir)
            .env("LABINIUS_LABRADOR_DIR", &native)
            .env("LABINIUS_DIAGNOSTIC_KEYDIR", outer)
            .env_remove("CARGO_MAKEFLAGS")
            .env_remove("MAKEFLAGS")
            .env_remove("MFLAGS")
            .output()
            .expect("run one exact correctness target");
        for (stream, bytes) in [("stdout", &output.stdout), ("stderr", &output.stderr)] {
            let path = work.join(format!("{case}.{stream}"));
            fs::write(&path, bytes).unwrap();
            record.artifact(case, stream, &path);
            for line in String::from_utf8_lossy(bytes).lines() {
                println!("candidate-c-correctness {case} {stream} {line}");
            }
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        let summaries: Vec<_> = stdout
            .lines()
            .filter(|l| l.starts_with("test result:"))
            .collect();
        let named = format!("test {test} ... ok");
        let one = summaries.len() == 1
            && summaries[0].starts_with("test result: ok. 1 passed; 0 failed;")
            && stdout.lines().filter(|l| *l == named).count() == 1;
        record.emit(
            "correctness",
            &[
                ("case", quote(case)),
                ("test", quote(test)),
                ("exit", output.status.code().unwrap_or(-1).to_string()),
                ("tests", if one { "1" } else { "0" }.into()),
                ("pass", (output.status.success() && one).to_string()),
                ("stdout", quote(&stdout)),
            ],
        );
        assert!(
            output.status.success() && one,
            "exact native correctness target failed"
        );
        let release = target_dir.join(&host).join("release");
        let stem = target.replace('-', "_") + "-";
        let mut binaries = Vec::new();
        for e in fs::read_dir(release.join("deps")).unwrap() {
            let p = e.unwrap().path();
            if p.file_name().unwrap().to_str().unwrap().starts_with(&stem)
                && p.extension().is_none()
            {
                binaries.push(p);
            }
        }
        assert_eq!(binaries.len(), 1);
        record.artifact(case, "test-executable", &binaries[0]);
        bindings(record, case, &helper_out(&release));
        native_link(record, case, &release, &native);
        if case == "focused" {
            let out = helper_out(&release);
            let check = out.join("mean-image-check.tsv");
            let text = fs::read_to_string(&check).unwrap();
            let f: Vec<_> = text.trim_end().split('\t').collect();
            assert_eq!(f.len(), 7);
            let raw = out.join("sms-mur.bin");
            let derived = out.join("canonical-mean-static.bin");
            assert_eq!(f[0], fs::metadata(&raw).unwrap().len().to_string());
            assert_eq!(f[1], sha256(&raw));
            assert_eq!(f[2], fs::metadata(&derived).unwrap().len().to_string());
            assert_eq!(f[3], sha256(&derived));
            assert_eq!(f[6], sha256(&out.join("mean-static.rs")));
            record.artifact(case, "mean-image-check", &check);
            record.artifact(case, "canonical-mean-static", &derived);
            record.emit("mean-image", &[
                ("raw_bytes", f[0].into()), ("raw_sha256", quote(f[1])),
                ("canonical_bytes", f[2].into()), ("canonical_sha256", quote(f[3])),
                ("entries", f[4].into()), ("alignment", f[5].into()),
                ("view_sha256", quote(f[6])),
            ]);
        }
    }
}

fn profile(stderr: &str) -> String {
    let lines: Vec<_> = stderr
        .lines()
        .filter(|l| l.starts_with("labinius-profile "))
        .collect();
    assert_eq!(lines.len(), 1);
    let fields: BTreeMap<_, _> = lines[0]
        .split_whitespace()
        .skip(1)
        .map(|s| s.split_once('=').unwrap())
        .collect();
    let names = [
        "load_us",
        "matrix_us",
        "setup_us",
        "commit_us",
        "recombine_us",
        "partials_us",
        "switch_us",
        "row_us",
        "challenges_us",
        "fold_us",
        "encode_us",
        "total_us",
        "threads",
        "fold_gfni",
        "sms_end_us",
        "sms_groups",
        "sms_decoupled",
        "sms_decoupled_at",
        "fold_end_us",
        "fold_tasks",
    ];
    assert_eq!(fields.len(), 20);
    assert_eq!(lines[0].split_whitespace().count(), 21);
    for name in names {
        assert!(fields.contains_key(name));
    }
    for name in &names[..13] {
        fields[name].parse::<u128>().unwrap();
    }
    assert_eq!(fields["threads"], "8");
    assert!(matches!(fields["fold_gfni"], "true" | "false"));
    for name in ["sms_end_us", "sms_groups", "fold_end_us", "fold_tasks"] {
        let values: Vec<_> = fields[name]
            .split('/')
            .map(|v| v.parse::<u128>().unwrap())
            .collect();
        assert_eq!(values.len(), 8);
    }
    let decoupled = fields["sms_decoupled"].parse::<usize>().unwrap();
    let at = fields["sms_decoupled_at"];
    let values: Vec<_> = if at.is_empty() {
        Vec::new()
    } else {
        at.split('/').collect()
    };
    assert_eq!(values.len(), decoupled.min(16));
    for value in values {
        let (worker, time) = value.split_once('@').unwrap();
        assert!(worker.parse::<usize>().unwrap() < 8);
        time.parse::<u128>().unwrap();
    }
    let clocks: Vec<_> = stderr
        .lines()
        .filter(|l| l.starts_with("labinius-clock "))
        .collect();
    assert_eq!(clocks.len(), 1);
    let clocks: BTreeMap<_, _> = clocks[0]
        .split_whitespace()
        .skip(1)
        .map(|v| v.split_once('=').unwrap())
        .collect();
    assert_eq!(clocks.len(), 2);
    assert!(
        clocks["exit_ns"].parse::<u128>().unwrap() >= clocks["main_ns"].parse::<u128>().unwrap()
    );
    let output: Vec<_> = stderr
        .lines()
        .filter(|l| l.starts_with("labinius-output "))
        .collect();
    assert_eq!(output.len(), 1);
    let output: BTreeMap<_, _> = output[0]
        .split_whitespace()
        .skip(1)
        .map(|v| v.split_once('=').unwrap())
        .collect();
    assert_eq!(output.len(), 2);
    for name in ["prove_return_us", "write_us"] {
        output[name].parse::<u128>().unwrap();
    }
    lines[0].to_owned()
}

fn child(
    record: &mut Record,
    bin: &Path,
    work: &Path,
    table: &Path,
    role: usize,
    call: usize,
    phase: &str,
    pair: usize,
    order: &str,
    position: usize,
    expected: &mut Option<Vec<u8>>,
) -> u128 {
    let proof = work.join(format!("proof-{call:02}.bin"));
    assert!(!proof.exists());
    let stdout_path = work.join(format!("child-{call:02}.stdout"));
    let stderr_path = work.join(format!("child-{call:02}.stderr"));
    let seed = "51".repeat(32);
    let mut command = Command::new(bin);
    command
        .args(["--prove"])
        .arg(&proof)
        .arg("--table")
        .arg(table)
        .current_dir(work)
        .env_clear()
        .env("HOME", "/tmp")
        .env("TMPDIR", "/tmp")
        .env("LABINIUS_SEED", &seed)
        .stdin(Stdio::null());
    record.emit(
        "command",
        &[
            ("call", call.to_string()),
            ("role", quote(ROLES[role])),
            ("executable", quote(bin.to_str().unwrap())),
            ("proof", quote(proof.to_str().unwrap())),
            ("table", quote(table.to_str().unwrap())),
            ("seed", quote(&seed)),
            ("environment", quote("HOME=/tmp;TMPDIR=/tmp;LABINIUS_SEED")),
            ("args", quote("--prove,proof,--table,table")),
        ],
    );
    // Keep this exact bracket: no logging or proof work inside the raw outer timer.
    let started = Instant::now();
    let output = command.output();
    let outer_ns = started.elapsed().as_nanos();
    let output = output.expect("fresh whole production child");
    fs::write(&stdout_path, &output.stdout).unwrap();
    fs::write(&stderr_path, &output.stderr).unwrap();
    record.artifact("children", &format!("child-{call:02}-stdout"), &stdout_path);
    record.artifact("children", &format!("child-{call:02}-stderr"), &stderr_path);
    let stderr = std::str::from_utf8(&output.stderr).unwrap();
    let actual_profile = profile(stderr);
    record.emit(
        "child",
        &[
            ("call", call.to_string()),
            ("role", quote(ROLES[role])),
            ("phase", quote(phase)),
            ("pair", pair.to_string()),
            ("order", quote(order)),
            ("position", position.to_string()),
            ("timer", quote("child-output-outer")),
            ("unit", quote("ns")),
            ("outer_ns", outer_ns.to_string()),
            ("exit", output.status.code().unwrap_or(-1).to_string()),
            ("profile", quote(&actual_profile)),
            ("stderr", quote(stderr)),
            ("stdout_sha256", quote(&sha256(&stdout_path))),
            ("stderr_sha256", quote(&sha256(&stderr_path))),
        ],
    );
    assert!(output.status.success(), "production child failed: {stderr}");
    assert!(fs::symlink_metadata(&proof).unwrap().file_type().is_file());
    let bytes = fs::read(&proof).unwrap();
    assert_eq!(bytes.len(), PROOF_BYTES);
    let modulus = u128::from_le_bytes(bytes[4..20].try_into().unwrap());
    assert_eq!(modulus, MODULUS);
    let reference = expected.is_none();
    if let Some(expected) = expected {
        assert_eq!(&bytes, expected, "complete proof byte guard call={call}");
    }
    let hash = sha256(&proof);
    fs::remove_file(&proof).unwrap();
    record.emit(
        "proof",
        &[
            ("call", call.to_string()),
            ("role", quote(ROLES[role])),
            ("reference", reference.to_string()),
            ("bytes", bytes.len().to_string()),
            ("modulus", quote(&modulus.to_string())),
            ("sha256", quote(&hash)),
            ("equal", (!reference).to_string()),
        ],
    );
    if reference {
        *expected = Some(bytes);
    }
    outer_ns
}

pub fn run() {
    assert!(
        is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx512vbmi")
            && is_x86_feature_detected!("avx512vbmi2")
            && is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("gfni")
    );
    assert_eq!(std::thread::available_parallelism().unwrap().get(), 8);
    let cpu = fs::read_to_string("/proc/cpuinfo").unwrap();
    let model = cpu
        .lines()
        .find(|l| l.starts_with("model name"))
        .unwrap()
        .split_once(':')
        .unwrap()
        .1
        .trim();
    assert!(model.contains("Xeon(R) Platinum 8488C"));
    let outer = PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("parent key output directory"),
    )
    .canonicalize()
    .unwrap();
    let work = outer.join("candidate-c-records");
    fs::create_dir(&work).expect("fresh record directory");
    let mut record = Record {
        seq: 0,
        file: fs::File::create(work.join("records.jsonl")).unwrap(),
    };
    record.emit(
        "begin",
        &[
            ("case", quote("candidate-c-whole-process")),
            ("cpu", quote(model)),
            ("workers", "8".into()),
            ("mode", "0".into()),
            ("scheduler", quote("production-main")),
            ("parent_pool", quote("absent")),
            ("table_bytes", TABLE_BYTES.to_string()),
            ("modulus", quote(&MODULUS.to_string())),
            ("calls", "38".into()),
            ("guards", "37".into()),
            ("pairs", "16".into()),
            (
                "rustflags",
                quote(&std::env::var("LABINIUS_DIAGNOSTIC_RUSTFLAGS").unwrap_or_default()),
            ),
            (
                "encoded_rustflags",
                quote(&std::env::var("LABINIUS_DIAGNOSTIC_ENCODED_RUSTFLAGS").unwrap_or_default()),
            ),
        ],
    );
    let directory = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .to_owned();
    production_native(&mut record, &outer);
    correctness(&mut record, &outer, &work);
    bindings(&mut record, "performance", &helper_out(&directory));
    native_link(
        &mut record,
        "performance",
        &directory,
        &outer.join("candidate-c-performance-labrador"),
    );
    record.artifact("performance", "driver", &std::env::current_exe().unwrap());
    record.artifact(
        "performance",
        "build-stdout",
        &outer.join("candidate-c-build.stdout"),
    );
    record.artifact(
        "performance",
        "build-stderr",
        &outer.join("candidate-c-build.stderr"),
    );
    let bins = [directory.join(ROLES[0]), directory.join(ROLES[1])];
    for (role, bin) in ROLES.iter().zip(&bins) {
        record.artifact("performance", role, bin);
    }

    let table = work.join("witness-table.bin");
    let mut file = fs::File::create(&table).unwrap();
    let mut input = blake3::Hasher::new_keyed(&[0x51u8; 32]);
    input.update(b"labinius-fast/binary-table/v1");
    let mut reader = input.finalize_xof();
    let mut block = vec![0u8; 1 << 20];
    for _ in 0..TABLE_BYTES / block.len() {
        reader.fill(&mut block);
        file.write_all(&block).unwrap();
    }
    drop(file);
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(&table, fs::Permissions::from_mode(0o444)).unwrap();
    record.artifact("performance", "table", &table);
    let mut expected = None;
    let mut call = 0;
    for role in 0..2 {
        call += 1;
        child(
            &mut record,
            &bins[role],
            &work,
            &table,
            role,
            call,
            if role == 0 { "reference" } else { "validation" },
            0,
            "",
            role + 1,
            &mut expected,
        );
    }
    for round in 0..18 {
        let order = if round % 2 == 0 { "AB" } else { "BA" };
        let mut times = [0u128; 2];
        for position in 0..2 {
            let role = if round % 2 == 0 {
                position
            } else {
                1 - position
            };
            call += 1;
            times[role] = child(
                &mut record,
                &bins[role],
                &work,
                &table,
                role,
                call,
                if round < 2 { "warmup" } else { "sample" },
                if round < 2 { round + 1 } else { round - 1 },
                order,
                position + 1,
                &mut expected,
            );
        }
        if round >= 2 {
            record.emit(
                "pair",
                &[
                    ("pair", (round - 1).to_string()),
                    ("order", quote(order)),
                    ("a_ns", times[0].to_string()),
                    ("b_ns", times[1].to_string()),
                    (
                        "delta_ns",
                        (times[0] as i128 - times[1] as i128).to_string(),
                    ),
                ],
            );
        }
    }
    assert_eq!(call, 38);
    fs::remove_file(&table).unwrap();
    record.emit(
        "complete",
        &[
            ("calls", "38".into()),
            ("guards", "37".into()),
            ("pairs", "16".into()),
            ("correctness_targets", "2".into()),
        ],
    );
}
