#[cfg(feature = "opencv-video")]
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
#[cfg(feature = "opencv-video")]
use std::time::Duration;

use rayon::{ThreadPool, ThreadPoolBuildError, ThreadPoolBuilder};

const MAX_PREPROCESS_THREADS: usize = 64;

/// Shared execution resources for multimodal decode and preprocessing.
///
/// A runtime is owned by the application-level multimodal components and is
/// independent of any model, modality, decoder implementation, or inference
/// backend. Keeping these resources explicit avoids process-global scheduling
/// state and lets all stages share the same CPU budget.
pub struct MultimodalRuntime {
    preprocess_pool: ThreadPool,
    #[cfg(feature = "opencv-video")]
    video_decodes: Arc<VideoDecodeScheduler>,
}

impl MultimodalRuntime {
    pub fn new() -> Result<Self, ThreadPoolBuildError> {
        let available_parallelism = std::thread::available_parallelism()
            .map(|parallelism| parallelism.get())
            .unwrap_or(1);
        let preprocess_threads = available_parallelism.min(MAX_PREPROCESS_THREADS);
        let preprocess_pool = ThreadPoolBuilder::new()
            .num_threads(preprocess_threads)
            .thread_name(|index| format!("smg-mm-preprocess-{index}"))
            .build()?;

        Ok(Self {
            preprocess_pool,
            #[cfg(feature = "opencv-video")]
            video_decodes: Arc::new(VideoDecodeScheduler {
                active: AtomicUsize::new(0),
                available_parallelism,
            }),
        })
    }

    /// Execute CPU-heavy multimodal work in this runtime's worker pool.
    pub fn run_cpu<OP, R>(&self, op: OP) -> R
    where
        OP: FnOnce() -> R + Send,
        R: Send,
    {
        self.preprocess_pool.install(op)
    }

    #[cfg(feature = "opencv-video")]
    pub(crate) fn enter_video_decode(&self, coalesce_window: Duration) -> ActiveVideoDecode {
        ActiveVideoDecode::enter(self.video_decodes.clone(), coalesce_window)
    }
}

#[cfg(feature = "opencv-video")]
struct VideoDecodeScheduler {
    active: AtomicUsize,
    available_parallelism: usize,
}

#[cfg(feature = "opencv-video")]
pub(crate) struct ActiveVideoDecode {
    scheduler: Arc<VideoDecodeScheduler>,
    observed: usize,
}

#[cfg(feature = "opencv-video")]
impl ActiveVideoDecode {
    fn enter(scheduler: Arc<VideoDecodeScheduler>, coalesce_window: Duration) -> Self {
        scheduler.active.fetch_add(1, Ordering::AcqRel);
        std::thread::sleep(coalesce_window);
        Self {
            observed: scheduler.active.load(Ordering::Acquire),
            scheduler,
        }
    }

    pub(crate) fn count(&self) -> usize {
        self.observed
    }

    pub(crate) fn available_parallelism(&self) -> usize {
        self.scheduler.available_parallelism
    }
}

#[cfg(feature = "opencv-video")]
impl Drop for ActiveVideoDecode {
    fn drop(&mut self) {
        self.scheduler.active.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_work_runs_in_owned_pool() {
        let Ok(runtime) = MultimodalRuntime::new() else {
            panic!("multimodal runtime should initialize");
        };

        let thread_name = runtime.run_cpu(|| {
            let thread = std::thread::current();
            thread.name().map(str::to_owned)
        });

        assert!(thread_name
            .as_deref()
            .is_some_and(|name| name.starts_with("smg-mm-preprocess-")));
    }

    #[cfg(feature = "opencv-video")]
    #[test]
    fn video_decode_state_is_scoped_to_runtime() {
        let Ok(first_runtime) = MultimodalRuntime::new() else {
            panic!("first multimodal runtime should initialize");
        };
        let Ok(second_runtime) = MultimodalRuntime::new() else {
            panic!("second multimodal runtime should initialize");
        };

        let first = first_runtime.enter_video_decode(Duration::ZERO);
        let concurrent = first_runtime.enter_video_decode(Duration::ZERO);
        let independent = second_runtime.enter_video_decode(Duration::ZERO);

        assert_eq!(first.count(), 1);
        assert_eq!(concurrent.count(), 2);
        assert_eq!(independent.count(), 1);

        drop(first);
        drop(concurrent);
        let next = first_runtime.enter_video_decode(Duration::ZERO);
        assert_eq!(next.count(), 1);
    }
}
