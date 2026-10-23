//! Optional self-only counting. No syscall is made inside the supplied timed closure.
use std::time::Instant;

const THREADS: usize = 8;
const EVENTS: usize = 2;
const WORDS: usize = 3 + 2 * EVENTS;
const RESET: u64 = 0x2403;
const ENABLE: u64 = 0x2400;
const DISABLE: u64 = 0x2401;

trait Backend {
    fn tasks(&mut self) -> Result<Vec<u32>, String>;
    fn open(&mut self, tid: u32, config: u64, leader: i32) -> Result<i32, String>;
    fn id(&mut self, fd: i32) -> Result<u64, String>;
    fn control(&mut self, fd: i32, op: u64) -> Result<(), String>;
    fn read(&mut self, fd: i32) -> Result<Vec<u64>, String>;
    fn close(&mut self, fd: i32);
}

#[derive(Clone, Debug)]
struct Counts { values: [u64; EVENTS], enabled: u64, running: u64 }

fn parse(words: &[u64], ids: &[u64; EVENTS]) -> Result<Counts, String> {
    if words.len() != WORDS || words[0] != EVENTS as u64 || words[2] > words[1] {
        return Err("group_read_format".into());
    }
    let mut values = [0; EVENTS];
    let mut seen = [false; EVENTS];
    for pair in words[3..].chunks_exact(2) {
        let index = ids.iter().position(|id| *id == pair[1]).ok_or("group_read_unknown_id")?;
        if seen[index] { return Err("group_read_duplicate_id".into()); }
        seen[index] = true;
        values[index] = pair[0];
    }
    Ok(Counts { values, enabled: words[1], running: words[2] })
}

fn validate_task_ids(mut tids: Vec<u32>, caller: u32) -> Result<Vec<u32>, String> {
    tids.sort_unstable();
    if tids.len() != THREADS || tids.iter().any(|tid| *tid == 0 || *tid > i32::MAX as u32)
        || tids.windows(2).any(|p| p[0] == p[1]) || !tids.contains(&caller) {
        return Err("self_task_count_ids_or_caller".into());
    }
    Ok(tids)
}
fn validate_status(text: &str, tid: u32, pid: u32) -> Result<(), String> {
    let field = |key: &str| -> Option<u32> {
        text.lines().find_map(|line| line.strip_prefix(key)?.trim().parse().ok())
    };
    if field("Tgid:") != Some(pid) || field("Pid:") != Some(tid) {
        return Err("self_task_owner".into());
    }
    Ok(())
}

struct CpuRecord { cpu_ns: u128, bracket_ns: u128 }
fn measure_cpu<T>(call: impl FnOnce() -> (T, u128), mut clock: impl FnMut() -> Result<u128, String>)
    -> ((T, u128), Result<CpuRecord, String>) {
    let bracket = Instant::now();
    let start = clock();
    let result = call();
    let end = clock();
    let bracket_ns = bracket.elapsed().as_nanos();
    let cpu = match (start, end) {
        (Ok(start), Ok(end)) => end.checked_sub(start).map(|cpu_ns| CpuRecord { cpu_ns, bracket_ns }).ok_or_else(|| "process_cpu_time_decreased".into()),
        (Err(start), Err(end)) => Err(format!("cpu_start_{start};cpu_end_{end}")),
        (Err(error), _) => Err(format!("cpu_start_{error}")),
        (_, Err(error)) => Err(format!("cpu_end_{error}")),
    };
    (result, cpu)
}
fn print_cpu(sample: usize, cpu: Result<CpuRecord, String>, elapsed_ns: u128) {
    match cpu {
        Ok(cpu) => println!("native-cpu sample={sample} status=AVAILABLE clock=CLOCK_PROCESS_CPUTIME_ID scope=all_self_threads_user_system cpu_ns={} cpu_bracket_ns={} api_elapsed_ns={elapsed_ns} fringe_ns={} fringe_valid={} boundary=sequential_reads", cpu.cpu_ns, cpu.bracket_ns, cpu.bracket_ns.saturating_sub(elapsed_ns), cpu.bracket_ns >= elapsed_ns),
        Err(error) => println!("native-cpu sample={sample} status=UNAVAILABLE clock=CLOCK_PROCESS_CPUTIME_ID scope=all_self_threads_user_system reason={error:?}"),
    }
}
fn cpu_nanos(sec: i64, nsec: i64) -> Result<u128, String> {
    if sec < 0 || !(0..1_000_000_000).contains(&nsec) { return Err("cpu_timespec_invalid".into()); }
    Ok(sec as u128 * 1_000_000_000 + nsec as u128)
}

struct Group { fds: Vec<i32>, ids: [u64; EVENTS] }
struct Session<B: Backend> {
    backend: B,
    tids: Vec<u32>,
    groups: Vec<Group>,
    enabled: bool,
}
impl<B: Backend> Session<B> {
    fn new(backend: B) -> Result<Self, String> {
        let mut session = Self { backend, tids: Vec::new(), groups: Vec::new(), enabled: false };
        session.tids = session.backend.tasks()?;
        if session.tids.len() != THREADS || session.tids.windows(2).any(|p| p[0] >= p[1]) {
            return Err("task_count_or_duplicate".into());
        }
        for index in 0..THREADS {
            session.groups.push(Group { fds: Vec::new(), ids: [0; EVENTS] });
            for event in 0..EVENTS {
                let leader = session.groups[index].fds.first().copied().unwrap_or(-1);
                let fd = session.backend.open(session.tids[index], event as u64, leader)?;
                session.groups[index].fds.push(fd);
                let id = session.backend.id(fd)?;
                if id == 0 || session.groups.iter().any(|g| g.ids.contains(&id)) {
                    return Err("event_id_zero_or_duplicate".into());
                }
                session.groups[index].ids[event] = id;
            }
        }
        session.check_tasks()?;
        Ok(session)
    }
    fn check_tasks(&mut self) -> Result<(), String> {
        if self.backend.tasks()? != self.tids { return Err("task_set_changed".into()); }
        Ok(())
    }
    fn snapshots(&mut self) -> Result<Vec<Counts>, String> {
        self.groups.iter().map(|g| parse(&self.backend.read(g.fds[0])?, &g.ids)).collect()
    }
    fn start(&mut self) -> Result<(Vec<Counts>, Instant, u128), String> {
        self.check_tasks()?;
        // RESET clears values, not cumulative time_enabled/time_running.
        let before = self.snapshots()?;
        for group in &self.groups { self.backend.control(group.fds[0], RESET)?; }
        let start = Instant::now();
        self.enabled = true; // cleanup must disable even after a partial enable failure
        for group in &self.groups { self.backend.control(group.fds[0], ENABLE)?; }
        let enable_ns = start.elapsed().as_nanos();
        Ok((before, start, enable_ns))
    }
    fn stop(&mut self) -> Result<u128, String> {
        let clock = Instant::now();
        let mut first = None;
        for group in &self.groups {
            if let Err(error) = self.backend.control(group.fds[0], DISABLE) {
                if first.is_none() { first = Some(error); }
            }
        }
        self.enabled = false; // closing FDs handles a failed disable; do not retry it
        first.map_or(Ok(clock.elapsed().as_nanos()), Err)
    }
}
impl<B: Backend> Drop for Session<B> {
    fn drop(&mut self) {
        if self.enabled { let _ = self.stop(); }
        for group in self.groups.iter().rev() {
            for &fd in group.fds.iter().rev() { self.backend.close(fd); }
        }
    }
}

struct Engine<B: Backend> { session: Option<Session<B>>, reason: Option<String> }
impl<B: Backend> Engine<B> {
    fn new(backend: B) -> Self {
        match Session::new(backend) {
            Ok(session) => {
                println!("native-pmu status=OPENED threads=8 events=user_cycles,user_instructions inherit=0 cpu=-1 exclude_kernel=1 exclude_hv=1 sampling=0 raw_events=OMITTED raw_reason=generic_only");
                for (index, tid) in session.tids.iter().enumerate() {
                    println!("native-pmu inventory=self thread_index={index} tid={tid}");
                }
                Self { session: Some(session), reason: None }
            }
            Err(error) => Self::unavailable(error),
        }
    }
    fn unavailable(reason: String) -> Self {
        println!("native-pmu status=UNAVAILABLE reason={reason:?} raw_events=OMITTED raw_reason=generic_only");
        Self { session: None, reason: Some(reason) }
    }
    fn measure_with_cpu<T>(&mut self, sample: usize, call: impl FnOnce() -> (T, u128), clock: impl FnMut() -> Result<u128, String>) -> (T, u128) {
        let mut cpu = None;
        let result = self.measure(sample, || {
            let (result, record) = measure_cpu(call, clock);
            cpu = Some(record);
            result
        });
        // Emit only after PMU stop/read. The complete output is still alive.
        print_cpu(sample, cpu.unwrap(), result.1);
        result
    }
    fn measure<T>(&mut self, sample: usize, call: impl FnOnce() -> (T, u128)) -> (T, u128) {
        let Some(mut session) = self.session.take() else {
            println!("native-pmu sample={sample} status=UNAVAILABLE reason={:?}", self.reason.as_deref().unwrap_or("unavailable"));
            return call();
        };
        let (before, start, enable_ns) = match session.start() {
            Ok(value) => value,
            Err(error) => {
                drop(session); // close every FD before running without counters
                self.reason = Some(error);
                println!("native-pmu sample={sample} status=UNAVAILABLE reason={:?}", self.reason.as_deref().unwrap_or("unavailable"));
                return call();
            }
        };
        // Taking Session out of self also closes/disables its FDs if call unwinds.
        let result = call();
        let stopped = session.stop();
        let bracket_ns = start.elapsed().as_nanos();
        let finish = (|| {
            let disable_ns = stopped?;
            session.check_tasks()?;
            let after = session.snapshots()?;
            let mut counts = Vec::with_capacity(THREADS);
            for (after, before) in after.into_iter().zip(&before) {
                let enabled = after.enabled.checked_sub(before.enabled).ok_or("enabled_time_decreased")?;
                let running = after.running.checked_sub(before.running).ok_or("running_time_decreased")?;
                if running > enabled || enabled == 0 || running == 0 {
                    return Err("counter_not_scheduled_or_invalid_time".into());
                }
                counts.push(Counts { enabled, running, ..after });
            }
            Ok::<_, String>((disable_ns, counts))
        })();
        match finish {
            Ok((disable_ns, counts)) => {
                let full = counts.iter().all(|c| c.running == c.enabled);
                let fringe = bracket_ns.checked_sub(result.1);
                println!("native-pmu sample={sample} status=AVAILABLE coverage={} threads=8 boundary=sequential counter_bracket_ns={bracket_ns} api_elapsed_ns={} enable_sweep_ns={enable_ns} disable_sweep_ns={disable_ns} fringe_ns={} fringe_valid={} time_fields=per_sample_delta raw_events=OMITTED", if full { "FULL" } else { "MULTIPLEXED" }, result.1, fringe.unwrap_or(0), fringe.is_some());
                let mut totals = [0u128; 4];
                for (index, counts) in counts.iter().enumerate() {
                    println!("native-pmu sample={sample} thread_index={index} tid={} user_cycles={} user_instructions={} time_enabled_ns={} time_running_ns={}", session.tids[index], counts.values[0], counts.values[1], counts.enabled, counts.running);
                    for (sum, value) in totals.iter_mut().zip([counts.values[0], counts.values[1], counts.enabled, counts.running]) { *sum += value as u128; }
                }
                println!("native-pmu sample={sample} aggregate=all8 user_cycles={} user_instructions={} time_enabled_ns={} time_running_ns={} scaling=none smt_scope=per_thread", totals[0], totals[1], totals[2], totals[3]);
                self.session = Some(session);
            }
            Err(error) => {
                drop(session);
                self.reason = Some(error);
                println!("native-pmu sample={sample} status=UNAVAILABLE reason={:?}", self.reason.as_deref().unwrap_or("unavailable"));
            }
        }
        result
    }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod linux {
    use super::*;
    use core::ffi::{c_int, c_long, c_ulong, c_void};
    extern "C" {
        fn getpid() -> c_int;
        fn syscall(number: c_long, ...) -> c_long;
        fn ioctl(fd: c_int, request: c_ulong, ...) -> c_int;
        fn read(fd: c_int, data: *mut c_void, count: usize) -> isize;
        fn close(fd: c_int) -> c_int;
        fn clock_gettime(clock: c_int, time: *mut Timespec) -> c_int;
    }
    #[repr(C)]
    struct Timespec { sec: c_long, nsec: c_long }
    const _: () = assert!(std::mem::size_of::<Timespec>() == 16 && std::mem::align_of::<Timespec>() == 8);
    pub(super) fn process_cpu() -> Result<u128, String> {
        let mut time = Timespec { sec: 0, nsec: 0 };
        if unsafe { clock_gettime(2, &mut time) } != 0 { return Err(os_error("process_cpu_clock")); }
        cpu_nanos(time.sec, time.nsec)
    }
    const SYS_GETTID: c_long = 186;
    const SYS_PERF_EVENT_OPEN: c_long = 298;
    #[repr(C)]
    struct Attr {
        kind: u32, size: u32, config: u64, sample_period: u64,
        sample_type: u64, read_format: u64, flags: u64,
        wakeup_events: u32, bp_type: u32, bp_addr: u64,
    }
    const _: () = assert!(std::mem::size_of::<Attr>() == 64);
    fn os_error(stage: &str) -> String {
        format!("{stage}_errno_{}", std::io::Error::last_os_error().raw_os_error().unwrap_or(-1))
    }
    pub(super) struct Linux { pid: u32, caller: u32 }
    impl Linux {
        pub(super) fn new() -> Self {
            Self { pid: unsafe { getpid() } as u32, caller: unsafe { syscall(SYS_GETTID) } as u32 }
        }
    }
    impl Backend for Linux {
        fn tasks(&mut self) -> Result<Vec<u32>, String> {
            let mut tids = Vec::new();
            for entry in std::fs::read_dir("/proc/self/task").map_err(|_| "task_list_read")? {
                let entry = entry.map_err(|_| "task_list_entry")?;
                let tid: u32 = entry.file_name().to_str().ok_or("task_name_utf8")?.parse().map_err(|_| "task_name_numeric")?;
                if tid == 0 || tid > i32::MAX as u32 { return Err("task_id_range".into()); }
                let text = std::fs::read_to_string(format!("/proc/self/task/{tid}/status")).map_err(|_| "self_task_status")?;
                validate_status(&text, tid, self.pid)?;
                tids.push(tid);
            }
            validate_task_ids(tids, self.caller)
        }
        fn open(&mut self, tid: u32, config: u64, leader: i32) -> Result<i32, String> {
            // PERF_ATTR_SIZE_VER0; hardware cycles=0/instructions=1; read GROUP|ID|times.
            let attr = Attr { kind: 0, size: 64, config, sample_period: 0, sample_type: 0,
                read_format: 15, flags: 1 | (1 << 5) | (1 << 6), wakeup_events: 0, bp_type: 0, bp_addr: 0 };
            let fd = unsafe { syscall(SYS_PERF_EVENT_OPEN, &attr, tid as c_int, -1i32, leader, 8 as c_ulong) };
            if fd < 0 { Err(os_error("perf_open")) } else { Ok(fd as i32) }
        }
        fn id(&mut self, fd: i32) -> Result<u64, String> {
            let mut id = 0u64;
            if unsafe { ioctl(fd, 0x8008_2407 as c_ulong, &mut id) } < 0 { Err(os_error("perf_id")) } else { Ok(id) }
        }
        fn control(&mut self, fd: i32, op: u64) -> Result<(), String> {
            if unsafe { ioctl(fd, op as c_ulong, 1 as c_ulong) } < 0 { Err(os_error(match op { ENABLE => "perf_enable", DISABLE => "perf_disable", RESET => "perf_reset", _ => "perf_control" })) } else { Ok(()) }
        }
        fn read(&mut self, fd: i32) -> Result<Vec<u64>, String> {
            let mut words = vec![0u64; WORDS];
            let bytes = unsafe { read(fd, words.as_mut_ptr().cast(), WORDS * 8) };
            if bytes < 0 { Err(os_error("perf_read")) }
            else if bytes != (WORDS * 8) as isize { Err("perf_read_short".into()) }
            else { Ok(words) }
        }
        fn close(&mut self, fd: i32) { unsafe { close(fd); } }
    }
}

/// Construct only after the helper's persistent pool and fixture setup are complete.
pub struct Probe {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    engine: Engine<linux::Linux>,
}
impl Probe {
    pub fn new() -> Self {
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        { Self { engine: Engine::new(linux::Linux::new()) } }
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
        { println!("native-pmu status=UNAVAILABLE reason=requires_linux_x86_64 raw_events=OMITTED"); Self {} }
    }
    /// call must run the unchanged timed API exactly once and return output + elapsed nanoseconds.
    pub fn measure<T>(&mut self, sample: usize, call: impl FnOnce() -> (T, u128)) -> (T, u128) {
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        { self.engine.measure_with_cpu(sample, call, linux::process_cpu) }
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
        {
            println!("native-pmu sample={sample} status=UNAVAILABLE reason=requires_linux_x86_64");
            let (result, cpu) = measure_cpu(call, || Err("requires_linux_x86_64".into()));
            print_cpu(sample, cpu, result.1);
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::{BTreeMap, BTreeSet};
    use std::rc::Rc;

    #[derive(Default)]
    struct State {
        opened: usize, closed: Vec<i32>, live: BTreeSet<i32>, groups: BTreeMap<i32, Vec<i32>>,
        enabled: BTreeSet<i32>, times: BTreeMap<i32, u64>, values: BTreeMap<i32, u64>,
        opens_fail_at: Option<usize>, ids_fail_at: Option<usize>, ids: usize,
        controls: usize, controls_fail_at: Option<usize>, reads: usize, reads_fail_at: Option<usize>,
        task_calls: usize, task_drift_at: Option<usize>, malformed_read_at: Option<usize>,
    }
    #[derive(Clone)] struct Mock(Rc<RefCell<State>>);
    impl Backend for Mock {
        fn tasks(&mut self) -> Result<Vec<u32>, String> {
            let mut s=self.0.borrow_mut();s.task_calls+=1;
            Ok((100..if s.task_drift_at==Some(s.task_calls) {107} else {108}).collect())
        }
        fn open(&mut self,_tid:u32,_config:u64,leader:i32)->Result<i32,String> {
            let mut s=self.0.borrow_mut();s.opened+=1;
            if s.opens_fail_at==Some(s.opened) {return Err("open_denied_errno_13".into());}
            let fd=s.opened as i32+10;assert!(s.live.insert(fd));
            if leader<0 {s.groups.insert(fd,vec![fd]);s.times.insert(fd,0);s.values.insert(fd,0);}
            else {s.groups.get_mut(&leader).unwrap().push(fd);}
            Ok(fd)
        }
        fn id(&mut self,fd:i32)->Result<u64,String> {
            let mut s=self.0.borrow_mut();s.ids+=1;
            if s.ids_fail_at==Some(s.ids) {Err("id_unsupported_errno_22".into())} else {Ok(fd as u64+1000)}
        }
        fn control(&mut self,fd:i32,op:u64)->Result<(),String> {
            let mut s=self.0.borrow_mut();s.controls+=1;
            if s.controls_fail_at==Some(s.controls) {return Err(format!("control_{op}_errno_5"));}
            match op {
                RESET=>{s.values.insert(fd,0);}
                ENABLE=>{s.enabled.insert(fd);}
                DISABLE=>{if s.enabled.remove(&fd) {*s.times.get_mut(&fd).unwrap()+=1000;*s.values.get_mut(&fd).unwrap()=100;}}
                _=>unreachable!(),
            }
            Ok(())
        }
        fn read(&mut self,fd:i32)->Result<Vec<u64>,String> {
            let mut s=self.0.borrow_mut();s.reads+=1;
            if s.reads_fail_at==Some(s.reads) {return Err("read_errno_5".into());}
            if s.malformed_read_at==Some(s.reads) {return Ok(vec![2,0]);}
            let fs=&s.groups[&fd];let t=s.times[&fd];let v=s.values[&fd];
            Ok(vec![2,t,t,v,fs[0] as u64+1000,v*2,fs[1] as u64+1000])
        }
        fn close(&mut self,fd:i32) {
            let mut s=self.0.borrow_mut();assert!(s.live.remove(&fd),"FD closed more than once");s.closed.push(fd);s.enabled.remove(&fd);
        }
    }
    fn state()->Rc<RefCell<State>> {Rc::new(RefCell::new(State::default()))}
    fn call_once(engine:&mut Engine<Mock>)->usize {
        let mut called=0;let result=engine.measure(1,||{called+=1;(123,0)});assert_eq!(result,(123,0));called
    }
    #[test] fn cpu_clock_success_failure_and_underflow_call_exactly_once() {
        assert_eq!(cpu_nanos(2,123).unwrap(),2_000_000_123);
        for (s,n) in [(-1,0),(0,-1),(0,1_000_000_000)] {assert!(cpu_nanos(s,n).is_err());}
        for answers in [[Ok(100),Ok(250)],[Err("blocked"),Ok(250)],[Ok(100),Err("blocked")],[Err("blocked"),Err("blocked")],[Ok(250),Ok(100)]] {
            let mut reads=0;let mut calls=0;
            let (result,cpu)=measure_cpu(||{calls+=1;(123,0)},||{let v=answers[reads].clone().map_err(String::from);reads+=1;v});
            assert_eq!(calls,1);assert_eq!(reads,2);assert_eq!(result,(123,0));
            if answers==[Ok(100),Ok(250)] {assert_eq!(cpu.unwrap().cpu_ns,150);} else {assert!(cpu.is_err());}
        }
    }
    #[test] fn cpu_clock_is_independent_of_perf_availability_for_all_five_calls() {
        for perf_denied in [false,true] {
            for cpu_denied in [false,true] {
                let state=state();if perf_denied {state.borrow_mut().opens_fail_at=Some(1);}
                let mut engine=Engine::new(Mock(state.clone()));let mut reads=0;let mut calls=0;
                for sample in 1..=5 {
                    let result=engine.measure_with_cpu(sample,||{calls+=1;(123,0)},||{reads+=1;if cpu_denied {Err("clock_denied_errno_1".into())} else {Ok(reads as u128*100)}});
                    assert_eq!(result,(123,0));
                }
                assert_eq!(reads,10);assert_eq!(calls,5);
                assert_eq!(engine.session.is_some(),!perf_denied);
                drop(engine);assert!(state.borrow().live.is_empty());
            }
        }
    }
    #[test] fn self_inventory_and_status_reject_foreign_or_missing_identity() {
        assert_eq!(validate_task_ids((100..108).rev().collect(),100).unwrap(),(100..108).collect::<Vec<_>>());
        for (tids,caller) in [(vec![100;8],100),((100..107).collect(),100),((100..108).collect(),999),(vec![0,1,2,3,4,5,6,7],1)] {
            assert!(validate_task_ids(tids,caller).is_err());
        }
        assert!(validate_status("Name: worker\nTgid:\t100\nPid:\t104\n",104,100).is_ok());
        for text in ["Tgid: 999\nPid: 104", "Tgid: 100\nPid: 999", "Tgid: 100", "Tgid: bad\nPid: 104"] {assert!(validate_status(text,104,100).is_err());}
    }
    #[test] fn parser_checks_length_ids_order_and_times() {
        assert_eq!(parse(&[2,20,10,7,12,9,11],&[11,12]).unwrap().values,[9,7]);
        for words in [vec![2,20,10,7,11,9,11],vec![2,20,10,7,13,9,12],vec![2,10,20,7,11,9,12],vec![1,20,10,7,11,9,12],vec![2,20,10]] {assert!(parse(&words,&[11,12]).is_err());}
    }
    #[test] fn complete_all8_and_five_calls_keep_time_deltas() {
        let state=state();let mut engine=Engine::new(Mock(state.clone()));
        for sample in 1..=5 {let result=engine.measure(sample,||([123,456],0));assert_eq!(result,([123,456],0));}
        assert!(engine.session.is_some());
        {let s=state.borrow();assert_eq!(s.live.len(),16);assert!(s.times.values().all(|t|*t==5000));assert!(s.enabled.is_empty());}
        drop(engine);assert!(state.borrow().live.is_empty());assert_eq!(state.borrow().closed.len(),16);
    }
    #[test] fn every_partial_open_and_id_failure_closes_all_fds() {
        for index in 1..=16 {
            for id_failure in [false,true] {
                let state=state();if id_failure {state.borrow_mut().ids_fail_at=Some(index);} else {state.borrow_mut().opens_fail_at=Some(index);}
                let mut engine=Engine::new(Mock(state.clone()));assert!(engine.session.is_none());
                for _ in 0..5 {assert_eq!(call_once(&mut engine),1);}
                assert!(state.borrow().live.is_empty());
                assert_eq!(state.borrow().closed.len(),if id_failure {index} else {index-1});
            }
        }
    }
    #[test] fn reset_enable_disable_failures_close_and_continue() {
        for index in 1..=24 {
            let state=state();state.borrow_mut().controls_fail_at=Some(index);
            let mut engine=Engine::new(Mock(state.clone()));assert_eq!(call_once(&mut engine),1);
            assert!(engine.session.is_none());assert!(state.borrow().live.is_empty());assert!(state.borrow().enabled.is_empty());
            for _ in 0..4 {assert_eq!(call_once(&mut engine),1);}
        }
    }
    #[test] fn every_read_error_or_malformed_read_closes_and_continues() {
        for index in 1..=16 {
            for malformed in [false,true] {
                let state=state();if malformed {state.borrow_mut().malformed_read_at=Some(index);} else {state.borrow_mut().reads_fail_at=Some(index);}
                let mut engine=Engine::new(Mock(state.clone()));assert_eq!(call_once(&mut engine),1);
                assert!(engine.session.is_none());assert!(state.borrow().live.is_empty());
                for _ in 0..4 {assert_eq!(call_once(&mut engine),1);}
            }
        }
    }
    #[test] fn task_drift_before_open_after_open_before_and_after_call() {
        for index in 1..=4 {
            let state=state();state.borrow_mut().task_drift_at=Some(index);
            let mut engine=Engine::new(Mock(state.clone()));assert_eq!(call_once(&mut engine),1);
            assert!(engine.session.is_none());assert!(state.borrow().live.is_empty());
        }
    }
    #[test] fn unwind_disables_and_closes_all_fds() {
        let state=state();let mut engine=Engine::new(Mock(state.clone()));
        let result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||engine.measure::<()>(1,||panic!("mock operation panic"))));
        assert!(result.is_err());assert!(engine.session.is_none());assert!(state.borrow().live.is_empty());assert!(state.borrow().enabled.is_empty());
    }
}
