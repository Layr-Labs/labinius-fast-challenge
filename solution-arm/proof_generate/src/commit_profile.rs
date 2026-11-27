//! Optional phase counts and timings for the promoted quarter-scale executor.
use crate::ntt::neon::sms;
use labinius::fields::scalar::B128;
use std::time::Instant;

pub(crate) const INDEX: usize = 0;
pub(crate) const LOOKUP: usize = 1;
pub(crate) const TAIL: usize = 2;
pub(crate) const SINK: usize = 3;
pub(crate) const CHUNK: usize = 4;
pub(crate) const BASE: usize = 5;
pub(crate) const RECONSTRUCT: usize = 6;
pub(crate) const REPLAY: usize = 7;
pub(crate) const REPAIR: usize = 8;
const LABELS: [&str; 9] = [
    "index", "lookup", "tail", "sink", "chunk", "base", "reconstruct", "replay", "repair",
];
const PERIOD: usize = 32;
const CLOCK_CALIBRATION: u64 = 512;

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn cpu_ns() -> Option<u64> {
    #[repr(C)]
    struct TimeSpec {
        sec: std::ffi::c_long,
        nsec: std::ffi::c_long,
    }
    extern "C" {
        fn clock_gettime(clock: std::ffi::c_int, out: *mut TimeSpec) -> std::ffi::c_int;
    }
    #[cfg(target_os = "linux")]
    const THREAD_CPU: std::ffi::c_int = 3;
    #[cfg(target_os = "macos")]
    const THREAD_CPU: std::ffi::c_int = 16;
    let mut value = TimeSpec { sec: 0, nsec: 0 };
    // AArch64 Linux and macOS use two C longs for timespec.
    (unsafe { clock_gettime(THREAD_CPU, &mut value) } == 0)
        .then(|| value.sec as u64 * 1_000_000_000 + value.nsec as u64)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn cpu_ns() -> Option<u64> { None }

#[derive(Default)]
pub(crate) struct WorkerProfile {
    pub first: usize,
    pub cols: usize,
    pub transforms: u64,
    pub primary_rows: u64,
    pub flag_slots: u64,
    pub flag_lanes: u64,
    pub exact_lanes: u64,
    pub parity_lanes: u64,
    pub parity_lifts: u64,
    elapsed: [u64; 9],
    counts: [u64; 9],
    clock_reads: u64,
    calibration_ns: u64,
    start_ns: u64,
    end_ns: u64,
    cpu_started: Option<u64>,
    cpu_elapsed: Option<u64>,
}
impl WorkerProfile {
    pub(crate) fn new(first: usize, dispatch: Instant) -> Self {
        let now = Instant::now();
        for _ in 0..CLOCK_CALIBRATION {
            std::hint::black_box(Instant::now());
        }
        let calibration_ns = now.elapsed().as_nanos() as u64;
        let cpu_started = cpu_ns();
        Self { first, calibration_ns, start_ns: dispatch.elapsed().as_nanos() as u64,
               cpu_started, ..Default::default() }
    }

    #[inline(always)]
    pub(crate) fn begin(&mut self) -> Instant {
        self.clock_reads += 1;
        Instant::now()
    }

    #[inline(always)]
    pub(crate) fn sample(&mut self, transform: usize, phase: usize) -> Option<Instant> {
        if (transform + phase) % PERIOD == 0 { Some(self.begin()) } else { None }
    }

    #[inline(always)]
    pub(crate) fn end(&mut self, phase: usize, before: Instant) -> Instant {
        let t = self.begin();
        self.elapsed[phase] += t.duration_since(before).as_nanos() as u64;
        self.counts[phase] += 1;
        t
    }

    #[inline(always)]
    pub(crate) fn split(&mut self, phase: usize, before: Option<Instant>) -> Option<Instant> {
        before.map(|t| self.end(phase, t))
    }

    pub(crate) fn finish(&mut self, dispatch: Instant) {
        self.end_ns = dispatch.elapsed().as_nanos() as u64;
        self.cpu_elapsed = self.cpu_started.zip(cpu_ns()).and_then(|(a,b)| b.checked_sub(a));
    }

    fn elapsed_ns(&self) -> u64 { self.end_ns - self.start_ns }

    fn adjusted(&self) -> [u64; 9] {
        std::array::from_fn(|i| {
            let count = self.counts[i];
            let x = self.elapsed[i].saturating_sub(self.calibration_ns * count / CLOCK_CALIBRATION);
            if i <= TAIL && count != 0 { x * self.transforms / count } else { x }
        })
    }

    fn print(&self, name: &str) {
        let adjusted = self.adjusted();
        let residual = self.elapsed_ns() as i64 - adjusted[..=RECONSTRUCT].iter().sum::<u64>() as i64;
        let cpu = self.cpu_elapsed.map_or_else(|| "unavailable".into(), |n| n.to_string());
        let estimate = self.calibration_ns * self.clock_reads / CLOCK_CALIBRATION;
        eprint!("diag_profile worker route={name} first={} cols={} worker_wall_ns={} elapsed_start_ns={} elapsed_end_ns={} thread_cpu_ns={cpu} transforms={} primary_rows={} flagged_slots={} flagged_lanes={} exact_lanes={} parity_lanes={} parity_lifts={} clock_reads={} clock_cal_calls={CLOCK_CALIBRATION} clock_cal_total_ns={} clock_read_estimate_ns={estimate} residual_estimate_ns={residual}",
                self.first,self.cols,self.elapsed_ns(),self.start_ns,self.end_ns,self.transforms,
                self.primary_rows,self.flag_slots,self.flag_lanes,self.exact_lanes,self.parity_lanes,
                self.parity_lifts,self.clock_reads,self.calibration_ns);
        for i in 0..LABELS.len() {
            let phase = LABELS[i];
            eprint!(" {phase}_observations={} {phase}_raw_ns={} {phase}_adjusted_ns={}",self.counts[i],self.elapsed[i],adjusted[i]);
        }
        eprintln!();
    }
}

fn profile_case(
    name: &str, reference: &[u8], unprofiled_ms: f64,
    commit: impl FnOnce(&mut [u8], Instant) -> Vec<WorkerProfile>,
) {
    let mut out = vec![0; reference.len()];
    for page in out.chunks_mut(4096) { page[0] = 1; }
    std::hint::black_box(&mut out);
    let start = Instant::now();
    let mut workers = commit(&mut out,start);
    let total_ns = start.elapsed().as_nanos() as u64;
    assert!(out == reference, "profiled commitment differs: {name}");
    workers.sort_by_key(|w| w.first);
    let critical = workers.iter().max_by_key(|w| w.end_ns).expect("profile workers");
    let cpu = workers.iter().try_fold(0u64, |total,w| w.cpu_elapsed.map(|n| total+n));
    let cpu = cpu.map_or_else(|| "unavailable".into(), |n| n.to_string());
    let estimates = workers.iter().map(WorkerProfile::adjusted).fold([0u64;9], |a,x| {
        std::array::from_fn(|i| a[i]+x[i])
    });
    for w in &workers { w.print(name); }
    let wall_sum: u64 = workers.iter().map(WorkerProfile::elapsed_ns).sum();
    let clock_sum: u64 = workers.iter().map(|w| w.clock_reads * w.calibration_ns / CLOCK_CALIBRATION).sum();
    let profile_minus_unprofiled_ms = total_ns as f64 / 1_000_000.0 - unprofiled_ms;
    eprint!("diag_profile end route={name} workers={} total_wall_ns={total_ns} aggregate_worker_wall_ns={wall_sum} aggregate_thread_cpu_ns={cpu} critical_first={} critical_worker_wall_ns={} critical_end_ns={} clock_read_estimate_sum_ns={clock_sum} unprofiled_reference_ms={unprofiled_ms:.6} profile_minus_reference_ms={profile_minus_unprofiled_ms:.6} bytes={} equal=true",
            workers.len(),critical.first,critical.elapsed_ns(),critical.end_ns,reference.len());
    for i in 0..LABELS.len() {
        let phase = LABELS[i];
        eprint!(" aggregate_{phase}_adjusted_ns={}",estimates[i]);
    }
    eprintln!();
}

pub(crate) fn run(values: &[B128], matrix: &sms::SmsMatrix, workers: usize, q: u128, bytes: usize) {
    eprintln!("diag_profile begin routes=1 period={PERIOD} workers={workers} clock=instant reference=quarter first_use=true hooks=false");
    let mut reference = vec![0; bytes];
    for page in reference.chunks_mut(4096) { page[0] = 1; }
    std::hint::black_box(&mut reference);
    let start = Instant::now();
    sms::commit_v::<17>(values, matrix, workers, q, &mut reference, false);
    let reference_ms = start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("diag_profile reference route=quarter wall_ms={reference_ms:.6} bytes={bytes}");
    profile_case("quarter", &reference, reference_ms, |out, start| {
        sms::key_profile::commit_profile::<17>(values, matrix, workers, q, out, false, start)
    });
    eprintln!("diag_profile close compared=true normal_proof_next=true");
}
