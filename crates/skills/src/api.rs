use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SkillServiceMode {
    #[default]
    Placeholder,
}

#[derive(Debug, Default)]
struct SkillServiceInner {
    mode: SkillServiceMode,
}

/// Placeholder service hook used to establish the app-context boundary before
/// the CRUD, storage, and resolution logic lands in later PRs.
#[derive(Debug, Clone, Default)]
pub struct SkillService {
    inner: Arc<SkillServiceInner>,
}

impl SkillService {
    pub fn placeholder() -> Self {
        Self {
            inner: Arc::new(SkillServiceInner {
                mode: SkillServiceMode::Placeholder,
            }),
        }
    }

    pub fn mode(&self) -> SkillServiceMode {
        self.inner.mode
    }
}

#[cfg(test)]
mod tests {
    use super::{SkillService, SkillServiceMode};

    #[test]
    fn placeholder_service_reports_placeholder_mode() {
        let service = SkillService::placeholder();
        assert_eq!(service.mode(), SkillServiceMode::Placeholder);
    }
}
