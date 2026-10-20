//! The little the parallel paths share: how many threads to use and how to cut a range.

/// `available_parallelism`, or 1.
pub fn threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// `n` items cut into at most `parts` contiguous ranges whose starts are multiples of `align`;
/// the last range takes the remainder. Empty ranges are not returned.
pub fn ranges(n: usize, parts: usize, align: usize) -> Vec<core::ops::Range<usize>> {
    let align = align.max(1);
    let parts = parts.max(1);
    let per = n.div_ceil(parts).div_ceil(align) * align;
    let mut out = Vec::with_capacity(parts);
    let mut start = 0;
    while start < n {
        let end = (start + per).min(n);
        out.push(start..end);
        start = end;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::ranges;

    #[test]
    fn ranges_cover_once() {
        for (n, parts, align) in [(0, 4, 64), (1, 8, 64), (1000, 8, 64), (4096, 3, 32), (5, 8, 1)] {
            let r = ranges(n, parts, align);
            assert!(r.len() <= parts);
            let mut at = 0;
            for x in &r {
                assert_eq!(x.start, at);
                assert!(x.start % align == 0);
                assert!(x.end > x.start);
                at = x.end;
            }
            assert_eq!(at, n);
        }
    }
}

// =============================================================================================
// a persistent pool
// =============================================================================================

/// The process's pool, `threads()` wide, made on first use. Never dropped: its workers park
/// between dispatches and go away with the process.
pub fn pool() -> &'static Pool {
    static POOL: std::sync::OnceLock<Pool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| Pool::new(threads()))
}
//
// `std::thread::scope` spawns a thread per task: a `clone`, a 2 MB stack mapping and its first
// faults, a join — tens of microseconds each, a few hundred times over a prover's phases. The
// pool below keeps `threads - 1` workers parked on a condvar and runs the caller's closure on
// them and on the calling thread, which is the same shape as a scope with one spawn per task
// but costs one wake-up per worker per dispatch.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// The job of one dispatch: the closure, its task count and the next task to hand out. The
/// closure pointer is only valid while [`Pool::run`] has not returned, and every worker drops it
/// before it counts itself out, so `run` returning is what ends its use.
struct Job {
    f: *const (dyn Fn(usize) + Sync),
    tasks: usize,
    next: AtomicUsize,
    /// Workers that have not yet finished with this job, the caller included.
    active: AtomicUsize,
}

unsafe impl Send for Job {}
unsafe impl Sync for Job {}

struct State {
    generation: u64,
    job: Option<Arc<Job>>,
    shutdown: bool,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    done: Condvar,
}

/// A fixed set of worker threads that run the tasks of one dispatch at a time.
pub struct Pool {
    shared: Arc<Shared>,
    /// Runners per dispatch: the workers plus the caller.
    width: usize,
    handles: Vec<std::thread::JoinHandle<()>>,
}

impl Pool {
    /// A pool of `width` runners: `width - 1` threads plus whoever calls [`run`](Self::run).
    pub fn new(width: usize) -> Pool {
        let width = width.max(1);
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                generation: 0,
                job: None,
                shutdown: false,
            }),
            wake: Condvar::new(),
            done: Condvar::new(),
        });
        let handles = (1..width)
            .map(|_| {
                let shared = shared.clone();
                std::thread::spawn(move || worker(&shared))
            })
            .collect();
        Pool {
            shared,
            width,
            handles,
        }
    }

    /// The pool's width: workers plus the caller.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Run `f(i)` for every `i` in `0 .. tasks`, each once, on the workers and the calling
    /// thread; returns once every task has finished. Tasks are handed out in order from a
    /// shared counter, so a slow task is not waited on by the others. `f` must not call back
    /// into this pool.
    pub fn run<F: Fn(usize) + Sync>(&self, tasks: usize, f: F) {
        if tasks == 0 {
            return;
        }
        if tasks == 1 || self.width == 1 {
            for i in 0..tasks {
                f(i);
            }
            return;
        }
        let f_ref: &(dyn Fn(usize) + Sync) = &f;
        // Extend the closure's lifetime for the workers; `run` does not return before every
        // worker has dropped its use of it (`active` reaches zero after their last call).
        let f_ptr: *const (dyn Fn(usize) + Sync) =
            unsafe { core::mem::transmute::<&(dyn Fn(usize) + Sync), &'static (dyn Fn(usize) + Sync)>(f_ref) };
        let job = Arc::new(Job {
            f: f_ptr,
            tasks,
            next: AtomicUsize::new(0),
            active: AtomicUsize::new(self.width),
        });
        {
            let mut state = self.shared.state.lock().unwrap();
            state.generation += 1;
            state.job = Some(job.clone());
            self.shared.wake.notify_all();
        }
        run_tasks(&job);
        if job.active.fetch_sub(1, Ordering::AcqRel) != 1 {
            let mut state = self.shared.state.lock().unwrap();
            while job.active.load(Ordering::Acquire) != 0 {
                state = self.shared.done.wait(state).unwrap();
            }
        }
        let mut state = self.shared.state.lock().unwrap();
        state.job = None;
    }

    /// `f(i)` for every `i` in `0 .. n`, collected in order.
    pub fn map<T: Send, F: Fn(usize) -> T + Sync>(&self, n: usize, f: F) -> Vec<T> {
        let mut out: Vec<core::mem::MaybeUninit<T>> = (0..n).map(|_| core::mem::MaybeUninit::uninit()).collect();
        let addr = out.as_mut_ptr() as usize;
        self.run(n, |i| unsafe {
            (addr as *mut core::mem::MaybeUninit<T>).add(i).write(core::mem::MaybeUninit::new(f(i)));
        });
        let mut out = core::mem::ManuallyDrop::new(out);
        unsafe { Vec::from_raw_parts(out.as_mut_ptr() as *mut T, out.len(), out.capacity()) }
    }
}

fn run_tasks(job: &Job) {
    loop {
        let i = job.next.fetch_add(1, Ordering::Relaxed);
        if i >= job.tasks {
            break;
        }
        unsafe { (*job.f)(i) };
    }
}

fn worker(shared: &Shared) {
    let mut seen = 0u64;
    loop {
        let job = {
            let mut state = shared.state.lock().unwrap();
            while state.generation == seen && !state.shutdown {
                state = shared.wake.wait(state).unwrap();
            }
            if state.shutdown {
                return;
            }
            seen = state.generation;
            state.job.clone()
        };
        let Some(job) = job else { continue };
        run_tasks(&job);
        if job.active.fetch_sub(1, Ordering::AcqRel) == 1 {
            let _state = shared.state.lock().unwrap();
            shared.done.notify_all();
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        {
            let mut state = self.shared.state.lock().unwrap();
            state.shutdown = true;
            self.shared.wake.notify_all();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod pool_tests {
    use super::*;

    #[test]
    fn every_task_runs_once_and_results_are_ordered() {
        let pool = Pool::new(5);
        for n in [0usize, 1, 3, 7, 64, 1000] {
            let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            pool.run(n, |i| {
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1), "n {n}");
            let squares = pool.map(n, |i| i * i);
            assert_eq!(squares, (0..n).map(|i| i * i).collect::<Vec<_>>());
        }
        // Many dispatches in a row, with a worker-dependent amount of work.
        let total = AtomicUsize::new(0);
        for round in 0..200 {
            pool.run(16, |i| {
                let mut x = 0usize;
                for k in 0..(i * 100) {
                    x = x.wrapping_add(k ^ round);
                }
                total.fetch_add(x.min(1) + 1, Ordering::Relaxed);
            });
        }
        assert_eq!(total.load(Ordering::Relaxed), 200 * 16 + 200 * 15);
    }
}
