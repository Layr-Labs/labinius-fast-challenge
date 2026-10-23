use std::{fs, io::Write, path::Path, process::Command};

const SEED: [u8; 32] = [0x51; 32];
const TABLE_BYTES: usize = 1 << 28;
const PROOF_BYTES: usize = 13_958_612;

fn bind_artifact(prefix: &str, kind: &str, path: &Path) -> String {
    let path = path.canonicalize().unwrap();
    let metadata = fs::symlink_metadata(&path).unwrap();
    assert!(metadata.file_type().is_file());
    let output = Command::new("/usr/bin/sha256sum").arg(&path).output().expect("hash actual runtime artifact");
    assert!(output.status.success());
    let text = std::str::from_utf8(&output.stdout).unwrap();
    let hash = text.split_whitespace().next().unwrap();
    assert!(hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit()));
    println!("{prefix} artifact kind={kind} path={} bytes={} sha256={hash}", path.display(), metadata.len());
    hash.to_owned()
}

fn bind_native_link(directory: &Path, outer: &Path, prefix: &str, target_name: &str) {
    let target = outer.join(target_name).canonicalize().unwrap();
    assert!(directory.starts_with(&target), "actual fresh helper target");
    let mut records = Vec::new();
    for entry in fs::read_dir(directory.join("build")).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_str().unwrap();
        if name.strip_prefix("labinius-").is_some_and(|s| s.len() == 16 && s.bytes().all(|b| b.is_ascii_hexdigit())) {
            let output = path.join("output");
            if output.is_file() {
                let text = fs::read_to_string(&output).unwrap();
                if text.lines().any(|s| s == "cargo:rustc-link-lib=static=labinius_c") { records.push((output, text)); }
            }
        }
    }
    assert_eq!(records.len(), 1, "one actual candidate native build output");
    let (metadata, text) = records.pop().unwrap();
    assert!(metadata.starts_with(&target));
    let libraries: Vec<_> = text.lines().filter(|s| s.starts_with("cargo:rustc-link-lib=")).collect();
    assert_eq!(libraries, ["cargo:rustc-link-lib=static=labinius_c", "cargo:rustc-link-lib=static=labrador", "cargo:rustc-link-lib=m"]);
    let searches: Vec<_> = text.lines().filter_map(|s| s.strip_prefix("cargo:rustc-link-search=native=")).map(Path::new).collect();
    assert_eq!(searches.len(), 2);
    let c_out = metadata.parent().unwrap().join("out").canonicalize().unwrap();
    let labrador = outer.join("native-diagnostics-labrador").canonicalize().unwrap();
    assert_eq!(searches[0].canonicalize().unwrap(), c_out);
    assert_eq!(searches[1].canonicalize().unwrap(), labrador);
    let c_archive = c_out.join("liblabinius_c.a");
    let labrador_archive = labrador.join("liblabrador.a");
    let stamp = labrador.join("libobj/.labinius_logq_stamp");
    assert_eq!(fs::read_to_string(&stamp).unwrap().trim(), "48");
    println!("{prefix} native-link target_root={} metadata_path={} c_archive_path={} labrador_archive_path={} logq=48", target.display(), metadata.display(), c_archive.display(), labrador_archive.display());
    bind_artifact(prefix, "native-link-metadata", &metadata);
    bind_artifact(prefix, "native-c-archive", &c_archive);
    bind_artifact(prefix, "native-labrador-archive", &labrador_archive);
    bind_artifact(prefix, "native-logq-stamp", &stamp);
    for line in text.lines().filter(|s| s.starts_with("cargo:rustc-link-search=") || s.starts_with("cargo:rustc-link-lib=")) {
        println!("{prefix} native-link-directive text={line}");
    }
}

fn write_table(path: &Path) {
    let mut hash = blake3::Hasher::new_keyed(&SEED);
    hash.update(b"labinius-fast/binary-table/v1");
    let mut reader = hash.finalize_xof();
    let mut file = fs::File::create(path).expect("create fixed seed-valid witness");
    let mut block = vec![0u8; 1 << 20];
    for _ in 0..TABLE_BYTES / block.len() {
        reader.fill(&mut block);
        file.write_all(&block).expect("write seed-valid witness");
    }
    drop(file);
    assert_eq!(fs::metadata(path).unwrap().len(), TABLE_BYTES as u64);
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o444)).unwrap();
}

fn child(bin: &Path, work: &Path, table: &Path, index: usize, ordinal: usize) -> (Vec<u8>, u128) {
    let proof = work.join("proof.bin");
    assert!(!proof.exists(), "no proof from an earlier child is reused");
    let seed: String = SEED.iter().map(|b| format!("{b:02x}")).collect();
    let mut command = Command::new(bin);
    command.args(["--prove"]).arg(&proof).arg("--table").arg(table)
        .env_clear().env("HOME", "/tmp").env("TMPDIR", "/tmp").env("LABINIUS_SEED", &seed)
        .current_dir(work);
    let name = crate::compare::NAMES[index];
    println!("native-context child-command call={ordinal} variant={name} executable_path={} work_path={} table_path={} proof_path={} seed_hex={seed} args=--prove,proof,--table,table env=HOME:/tmp,TMPDIR:/tmp,LABINIUS_SEED stdio=stdin-closed,stdout-captured,stderr-captured", bin.display(), work.display(), table.display(), proof.display());
    let started = std::time::Instant::now();
    let output = command.output();
    let outer_ns = started.elapsed().as_nanos();
    let exit = match &output {
        Ok(output) => output.status.code().map(|n| n.to_string()).unwrap_or_else(|| "signal".to_owned()),
        Err(_) => "io-error".to_owned(),
    };
    println!("native-context child call={ordinal} variant={name} outer_ns={outer_ns} exit={exit}");
    if let Err(error) = &output {
        println!("native-context failure call={ordinal} variant={name} io_error={error:?}");
    }
    let output = output.expect("run fresh actual prover child");
    fs::write(work.join(format!("call-{ordinal:02}-{name}.stdout")), &output.stdout).unwrap();
    fs::write(work.join(format!("call-{ordinal:02}-{name}.stderr")), &output.stderr).unwrap();
    let stderr = std::str::from_utf8(&output.stderr).expect("production diagnostic UTF-8");
    for line in stderr.lines() {
        println!("native-context raw call={ordinal} variant={name} stream=stderr text={line}");
    }
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        println!("native-context raw call={ordinal} variant={name} stream=stdout text={line}");
    }
    assert!(output.status.success(), "actual child failed call={ordinal}: {stderr}");
    let profile = crate::compare::profile(stderr);
    let (main_ns, exit_ns) = crate::compare::clock_ns(stderr);
    let main_elapsed_ns = exit_ns - main_ns;
    assert!(fs::symlink_metadata(&proof).unwrap().file_type().is_file());
    let bytes = fs::read(&proof).expect("read complete actual child proof");
    assert_eq!(bytes.len(), PROOF_BYTES, "complete fixed-shape proof");
    fs::remove_file(&proof).unwrap();
    println!("native-context proof call={ordinal} variant={name} proof_bytes={} total_ns={} load_us={} main_ns={main_ns} exit_ns={exit_ns} main_elapsed_ns={main_elapsed_ns} advice_status=UNOBSERVED", bytes.len(), profile.total_ns, profile.load_us);
    (bytes, outer_ns)
}

fn correctness(outer: &Path, work: &Path) {
    const TEST: &str = "tests::round_trip_and_reject_wrong_instance";
    let helper = Path::new(env!("CARGO_MANIFEST_DIR")).canonicalize().unwrap();
    let source = helper.parent().unwrap().join("proof_generate/src/main.rs");
    let manifest = helper.join("Cargo.toml");
    let target = outer.join("native-diagnostics-correctness-target");
    assert!(!target.exists(), "one fresh correctness target; no passing rerun");
    let native = outer.join("native-diagnostics-labrador");
    let key = outer.join("pair-key.bin");
    let cargo = std::env::var_os("CARGO").expect("actual inherited Cargo program");
    let host = std::env::var("HOST").expect("actual inherited native host");
    assert!(host.starts_with("x86_64-") && host.contains("linux"));
    let flags = std::env::var("RUSTFLAGS").unwrap_or_default();
    let encoded = std::env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    bind_artifact("native-correctness", "source", &source);
    bind_artifact("native-correctness", "manifest", &manifest);
    bind_artifact("native-correctness", "cargo", Path::new(&cargo));
    let version = Command::new(&cargo).arg("--version").output().expect("bind actual Cargo version");
    fs::write(work.join("native-correctness.cargo-version.stdout"), &version.stdout).unwrap();
    fs::write(work.join("native-correctness.cargo-version.stderr"), &version.stderr).unwrap();
    println!("native-correctness cargo-version exit={} text={:?}", version.status.code().unwrap_or(-1), String::from_utf8_lossy(&version.stdout));
    assert!(version.status.success());
    let args = vec!["test".to_owned(), "--locked".to_owned(), "--offline".to_owned(),
        "--release".to_owned(), "--jobs".to_owned(), "1".to_owned(), "--target".to_owned(), host.clone(),
        "-vv".to_owned(), "--manifest-path".to_owned(), manifest.to_str().unwrap().to_owned(),
        "--test".to_owned(), "prover-round-trip".to_owned(), TEST.to_owned(), "--".to_owned(),
        "--exact".to_owned(), "--nocapture".to_owned(), "--test-threads=1".to_owned()];
    let mut command = Command::new(&cargo);
    command.args(&args).current_dir(&helper)
        .env("CARGO_TARGET_DIR", &target).env("LABINIUS_LABRADOR_DIR", &native)
        .env("LABINIUS_DIAGNOSTIC_PAIR_KEY", &key)
        .env("LABINIUS_DIAGNOSTIC_RUSTFLAGS", &flags)
        .env("LABINIUS_DIAGNOSTIC_ENCODED_RUSTFLAGS", &encoded)
        .env_remove("CARGO_MAKEFLAGS").env_remove("MAKEFLAGS").env_remove("MFLAGS");
    let record = format!("native-correctness command program={cargo:?} argv={args:?} cwd={helper:?} target_dir={target:?} native_dir={native:?} pair_key={key:?} rustflags={flags:?} encoded_rustflags={encoded:?} remove_env=CARGO_MAKEFLAGS,MAKEFLAGS,MFLAGS manual_OUT_DIR=false\n");
    fs::write(work.join("native-correctness.command"), &record).unwrap();
    print!("{record}");
    let output = command.output();
    let output = match output {
        Ok(output) => output,
        Err(error) => {
            fs::write(work.join("native-correctness.stdout"), []).unwrap();
            fs::write(work.join("native-correctness.stderr"), format!("{error:?}\n")).unwrap();
            let result = format!("native-correctness result test={TEST} exit=io-error tests=UNOBSERVED outcome=FAIL error={error:?}\n");
            fs::write(work.join("native-correctness.result"), &result).unwrap();
            print!("{result}");
            panic!("existing native correctness command failed");
        }
    };
    fs::write(work.join("native-correctness.stdout"), &output.stdout).unwrap();
    fs::write(work.join("native-correctness.stderr"), &output.stderr).unwrap();
    for (stream, bytes) in [("stdout", &output.stdout), ("stderr", &output.stderr)] {
        for line in String::from_utf8_lossy(bytes).lines() {
            println!("native-correctness raw stream={stream} text={line}");
        }
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let summary_ok = text.lines().filter(|s| s.starts_with("test result:")).count() == 1
        && text.contains("test result: ok. 1 passed; 0 failed;");
    let named_ok = text.lines().filter(|s| *s == format!("test {TEST} ... ok")).count() == 1;
    let exit = output.status.code().map(|n| n.to_string()).unwrap_or_else(|| "signal".to_owned());
    let process = format!("native-correctness process test={TEST} exit={exit} summary_ok={summary_ok} named_ok={named_ok}\n");
    fs::write(work.join("native-correctness.process"), &process).unwrap();
    print!("{process}");
    if !output.status.success() || !summary_ok || !named_ok {
        let result = format!("native-correctness result test={TEST} exit={exit} tests=UNOBSERVED outcome=FAIL\n");
        fs::write(work.join("native-correctness.result"), &result).unwrap();
        print!("{result}");
        panic!("existing full-shape correctness test did not pass once");
    }
    let directory = target.join(&host).join("release");
    bind_native_link(&directory, outer, "native-correctness", "native-diagnostics-correctness-target");
    let outputs: Vec<_> = fs::read_dir(directory.join("build")).unwrap().map(|e| e.unwrap().path())
        .filter(|p| p.file_name().unwrap().to_str().unwrap().starts_with("labinius-native-diagnostics-")
            && p.join("out/context-bindings.txt").is_file()).collect();
    assert_eq!(outputs.len(), 1, "one actual correctness helper output");
    let out = outputs[0].join("out");
    bind_artifact("native-correctness", "generated-bindings", &out.join("context-bindings.txt"));
    for name in ["key.bin", "tail-key.bin", "hybrid-key.bin", "pair-key.bin"] {
        let parent_hash = bind_artifact("native-correctness", &format!("parent-{name}"), &outer.join(name));
        let test_hash = bind_artifact("native-correctness", &format!("test-{name}"), &out.join(name));
        assert_eq!(parent_hash, test_hash, "real correctness key image");
    }
    for name in ["native-correctness.command", "native-correctness.stdout", "native-correctness.stderr", "native-correctness.process"] {
        bind_artifact("native-correctness", name, &work.join(name));
    }
    let result = format!("native-correctness result test={TEST} exit=0 tests=1 passed=1 failed=0 outcome=PASS\n");
    fs::write(work.join("native-correctness.result"), &result).unwrap();
    print!("{result}");
}

pub fn run() {
    for line in include_str!(concat!(env!("OUT_DIR"), "/context-bindings.txt")).lines() {
        println!("native-context binding {line}");
    }
    let argument = std::env::args().nth(1).expect("accepted helper work path");
    let work = Path::new(&argument).with_extension(format!("context-{}", std::process::id()));
    fs::create_dir(&work).expect("fresh diagnostic work directory");
    let work = work.canonicalize().unwrap();
    let outer = Path::new(&argument).parent().unwrap().canonicalize().unwrap();
    correctness(&outer, &work);
    let table = work.join("witness-table.bin");
    write_table(&table);
    bind_artifact("native-context", "witness", &table);
    let directory = std::env::current_exe().unwrap().parent().unwrap().to_owned();
    let bins = [directory.join("prefix-context-baseline").canonicalize().unwrap(), directory.join("prefix-context-candidate").canonicalize().unwrap()];
    assert!(bins.iter().all(|p| p.is_file()), "both actual caller binaries exist");
    bind_artifact("native-context", "baseline-executable", &bins[0]);
    bind_artifact("native-context", "candidate-executable", &bins[1]);
    bind_native_link(&directory, &outer, "native-context", "native-diagnostics-target");
    println!("native-context config calls=38 processes=fresh complete_provers=true table_bytes={TABLE_BYTES} seed_byte=81 input=seed-valid real_commitment=true parent_pool=absent scheduler=actual-main advice_status=UNOBSERVED no_cost_subtraction=true");
    println!("native-context metric=raw-child-outer-ns precision_loss_lt_ns=0 rule=strict-positive-outer phase_agreement=NOT_APPLICABLE source_context_consistency=required");
    let mut call = |index: usize, ordinal: usize| child(bins[index].as_path(), &work, &table, index, ordinal);
    crate::compare::run(&mut call);
    fs::remove_file(&table).unwrap();
    println!("native-context phase=complete full_provers=38 guards=37 raw_logs={}", work.display());
}
