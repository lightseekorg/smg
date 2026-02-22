use std::{collections::HashMap, sync::Arc};

use tokio::task::JoinHandle;

use super::{
    error::{MultiModalError, MultiModalResult},
    media::{ImageFetchConfig, MediaConnector, MediaSource},
    types::{
        ChatContentPart, ImageDetail, Modality, MultiModalData, MultiModalUUIDs, TrackedMedia,
    },
};

type PendingTask = JoinHandle<MultiModalResult<TrackedMedia>>;

#[derive(Debug)]
pub struct TrackerOutput {
    pub data: MultiModalData,
    pub uuids: MultiModalUUIDs,
}

pub struct AsyncMultiModalTracker {
    media_connector: Arc<MediaConnector>,
    pending: HashMap<Modality, Vec<PendingTask>>,
    uuids: MultiModalUUIDs,
}

impl AsyncMultiModalTracker {
    pub fn new(media_connector: Arc<MediaConnector>) -> Self {
        Self {
            media_connector,
            pending: HashMap::new(),
            uuids: HashMap::new(),
        }
    }

    pub fn push_part(&mut self, part: ChatContentPart) -> MultiModalResult<()> {
        match part {
            ChatContentPart::Text { .. } => {}
            ChatContentPart::ImageUrl { url, detail, uuid } => {
                let source = match url::Url::parse(&url) {
                    Ok(parsed) if parsed.scheme() == "data" => MediaSource::DataUrl(url),
                    _ => MediaSource::Url(url),
                };
                self.enqueue_image(source, detail.unwrap_or_default(), uuid);
            }
            ChatContentPart::ImageData {
                data,
                mime_type: _,
                uuid,
                detail,
            } => {
                self.enqueue_image(
                    MediaSource::InlineBytes(data),
                    detail.unwrap_or_default(),
                    uuid,
                );
            }
            ChatContentPart::ImageEmbeds { .. } => {
                return Err(MultiModalError::UnsupportedContent("image_embeds"));
            }
        }
        Ok(())
    }

    pub async fn finalize(mut self) -> MultiModalResult<TrackerOutput> {
        let mut data = MultiModalData::new();
        for (modality, tasks) in self.pending.drain() {
            let mut items = Vec::with_capacity(tasks.len());
            for task in tasks {
                let media = task.await??;
                items.push(media);
            }
            data.insert(modality, items);
        }

        Ok(TrackerOutput {
            data,
            uuids: self.uuids,
        })
    }

    fn enqueue_image(&mut self, source: MediaSource, detail: ImageDetail, uuid: Option<String>) {
        let modality = Modality::Image;
        self.uuids.entry(modality).or_default().push(uuid);

        let connector = Arc::clone(&self.media_connector);
        #[expect(
            clippy::disallowed_methods,
            reason = "spawn handle is stored in self.pending and awaited in finalize(); fire-and-forget is intentional for concurrent media fetching"
        )]
        let handle = tokio::spawn(async move {
            let frame = connector
                .fetch_image(source, ImageFetchConfig { detail })
                .await?;
            Ok(TrackedMedia::Image(frame))
        });

        self.pending.entry(modality).or_default().push(handle);
    }
}
