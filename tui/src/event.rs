use crossterm::event::{Event, EventStream, KeyEvent, KeyEventKind};
use futures::StreamExt;
use tokio::{
    sync::mpsc,
    time::{interval, Duration},
};

/// Application-level events fed into the main loop.
#[derive(Debug)]
pub enum AppEvent {
    /// A key was pressed.
    Key(KeyEvent),
    /// Periodic tick for UI refresh.
    Tick,
    /// Terminal was resized.
    Resize(u16, u16),
}

/// Merges crossterm input events with a periodic tick timer.
pub struct EventHandler {
    rx: mpsc::UnboundedReceiver<AppEvent>,
}

impl EventHandler {
    /// Create and start the event handler.
    ///
    /// `tick_ms` controls how often [`AppEvent::Tick`] fires (drives UI refresh).
    pub fn new(tick_ms: u64) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();

        // Safety: fire-and-forget event reader loop that runs for the app's lifetime
        #[expect(clippy::disallowed_methods)]
        tokio::spawn(async move {
            let mut reader = EventStream::new();
            let mut tick = interval(Duration::from_millis(tick_ms));

            loop {
                tokio::select! {
                    _ = tick.tick() => {
                        if tx.send(AppEvent::Tick).is_err() {
                            break;
                        }
                    }
                    maybe_event = reader.next() => {
                        match maybe_event {
                            Some(Ok(Event::Key(key)))
                                if key.kind == KeyEventKind::Press
                                    && tx.send(AppEvent::Key(key)).is_err() =>
                            {
                                break;
                            }
                            Some(Ok(Event::Resize(w, h)))
                                if tx.send(AppEvent::Resize(w, h)).is_err() => {
                                    break;
                                }
                            Some(Err(_)) | None => break,
                            _ => {} // ignore mouse, focus, paste events
                        }
                    }
                }
            }
        });

        Self { rx }
    }

    /// Wait for the next event. Returns `None` if the channel closed.
    pub async fn next(&mut self) -> Option<AppEvent> {
        self.rx.recv().await
    }
}
