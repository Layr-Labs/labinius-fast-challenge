use std::{fs, io::Write, path::Path, process::Command};

const SEED: [u8; 32] = [0x51; 32];
const TABLE_BYTES: usize = 1 << 28;
const PROOF_BYTES: usize = 13_958_612;

fn bind_artifact(kind: &str, path: &Path) {
    let metadata = fs::symlink_metadata(path).unwrap();
    assert!(metadata.file_type().is_file());
    let output = Command::new("/usr/bin/sha256sum").arg(path).output().expect("hash actual runtime artifact");
    assert!(output.status.success());
    let text = std::str::from_utf8(&output.stdout).unwrap();
    let hash = text.split_whitespace().next().unwrap();
    assert!(hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit()));
    println!("native-context artifact kind={kind} bytes={} sha256={hash}", metadata.len());
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
    let output = Command::new(bin)
        .args(["--prove"]).arg(&proof).arg("--table").arg(table)
        .env_clear().env("HOME", "/tmp").env("TMPDIR", "/tmp").env("LABINIUS_SEED", seed)
        .current_dir(work).output().expect("run fresh actual prover child");
    let name = crate::compare::NAMES[index];
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
    let ns = crate::compare::phase_ns(stderr);
    assert!(fs::symlink_metadata(&proof).unwrap().file_type().is_file());
    let bytes = fs::read(&proof).expect("read complete actual child proof");
    assert_eq!(bytes.len(), PROOF_BYTES, "complete fixed-shape proof");
    fs::remove_file(&proof).unwrap();
    println!("native-context child call={ordinal} variant={name} exit=0 proof_bytes={} phase_ns={ns} phase_precision=integer-us-fields advice_status=UNOBSERVED", bytes.len());
    (bytes, ns)
}

pub fn run() {
    for line in include_str!(concat!(env!("OUT_DIR"), "/context-bindings.txt")).lines() {
        println!("native-context binding {line}");
    }
    let argument = std::env::args().nth(1).expect("accepted helper work path");
    let work = Path::new(&argument).with_extension(format!("context-{}", std::process::id()));
    fs::create_dir(&work).expect("fresh diagnostic work directory");
    let work = work.canonicalize().unwrap();
    let table = work.join("witness-table.bin");
    write_table(&table);
    bind_artifact("witness", &table);
    let directory = std::env::current_exe().unwrap().parent().unwrap().to_owned();
    let bins = [directory.join("prefix-context-baseline"), directory.join("prefix-context-candidate")];
    assert!(bins.iter().all(|p| p.is_file()), "both actual caller binaries exist");
    bind_artifact("baseline-executable", &bins[0]);
    bind_artifact("candidate-executable", &bins[1]);
    println!("native-context config calls=38 processes=fresh complete_provers=true table_bytes={TABLE_BYTES} seed_byte=81 input=seed-valid real_commitment=true parent_pool=absent scheduler=actual-main advice_status=UNOBSERVED no_cost_subtraction=true");
    println!("native-context metric=each-child-partials_us-plus-switch_us precision_loss_lt_ns=2000 rule=unchanged-phase-only calibration=required");
    let mut call = |index: usize, ordinal: usize| child(bins[index].as_path(), &work, &table, index, ordinal);
    crate::compare::run(&mut call);
    fs::remove_file(&table).unwrap();
    println!("native-context phase=complete full_provers=38 guards=37 raw_logs={}", work.display());
}
