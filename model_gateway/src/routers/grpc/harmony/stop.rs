//! Incremental stop-string scanning for harmony channel text.
//!
//! The direct-ZMQ engine receives token ids only and cannot match string
//! `stop` sequences; the regular pipeline covers this with the router-side
//! `StopSequenceDecoder`, but harmony emits channel-parsed text rather than
//! raw tokens, so stops are matched here on the decoded text instead. The
//! matched stop stays in the output as its suffix, mirroring what the
//! token-forwarding gRPC path yields for harmony models.

/// Result of feeding text through the scanner.
pub(crate) struct StopScan {
    /// Text safe to emit: everything up to and including a match, or the
    /// prefix that cannot be part of a future match.
    pub emit: String,
    /// A stop sequence completed inside this push.
    pub stopped: bool,
}

pub(crate) struct TextStopScanner {
    stops: Vec<String>,
    /// Tail held back because it could still grow into a match.
    pending: String,
    matched: Option<String>,
}

impl TextStopScanner {
    pub fn new(stops: Vec<String>) -> Self {
        Self {
            stops,
            pending: String::new(),
            matched: None,
        }
    }

    pub fn matched(&self) -> Option<&str> {
        self.matched.as_deref()
    }

    /// Feed a text delta. Once a stop has matched, further pushes emit nothing.
    pub fn push(&mut self, text: &str) -> StopScan {
        if self.matched.is_some() {
            return StopScan {
                emit: String::new(),
                stopped: true,
            };
        }
        self.pending.push_str(text);

        // Earliest match across all stops wins; ties prefer the longer stop.
        let mut best: Option<(usize, &str)> = None;
        for stop in &self.stops {
            if let Some(at) = self.pending.find(stop.as_str()) {
                let better = match best {
                    None => true,
                    Some((best_at, best_stop)) => {
                        at < best_at || (at == best_at && stop.len() > best_stop.len())
                    }
                };
                if better {
                    best = Some((at, stop));
                }
            }
        }
        if let Some((at, stop)) = best {
            let emit = self.pending[..at + stop.len()].to_string();
            self.matched = Some(stop.to_string());
            self.pending.clear();
            return StopScan {
                emit,
                stopped: true,
            };
        }

        // Hold back the longest suffix that is a prefix of some stop; emit the
        // rest (it can never become part of a match).
        let max_hold = (self.stops.iter().map(String::len).max())
            .unwrap_or(1)
            .saturating_sub(1);
        let mut hold = 0;
        for len in (1..=max_hold.min(self.pending.len())).rev() {
            let Some(start) = self.pending.len().checked_sub(len) else {
                continue;
            };
            if !self.pending.is_char_boundary(start) {
                continue;
            }
            let tail = &self.pending[start..];
            if self.stops.iter().any(|s| s.starts_with(tail)) {
                hold = len;
                break;
            }
        }
        let emit = self.pending[..self.pending.len() - hold].to_string();
        self.pending.drain(..self.pending.len() - hold);
        StopScan {
            emit,
            stopped: false,
        }
    }

    /// Emit whatever is still held back (end of stream, no match).
    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }

    /// Scan a complete text: returns the (possibly truncated) text and whether
    /// a stop matched inside it.
    pub fn scan_complete(&mut self, text: &str) -> (String, bool) {
        let scan = self.push(text);
        let mut out = scan.emit;
        if !scan.stopped {
            out.push_str(&self.flush());
        }
        (out, scan.stopped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_scan_truncates_after_first_match() {
        let mut scanner = TextStopScanner::new(vec![",".to_string()]);
        let (out, stopped) = scanner.scan_complete("1, 2, 3");
        assert_eq!(out, "1,");
        assert!(stopped);
        assert_eq!(scanner.matched(), Some(","));
    }

    #[test]
    fn streaming_holds_back_partial_matches() {
        let mut scanner = TextStopScanner::new(vec!["STOP".to_string()]);
        let scan = scanner.push("abc ST");
        assert_eq!(scan.emit, "abc ");
        assert!(!scan.stopped);
        let scan = scanner.push("OP tail");
        assert_eq!(scan.emit, "STOP");
        assert!(scan.stopped);
        // Post-match pushes are swallowed.
        let scan = scanner.push("more");
        assert_eq!(scan.emit, "");
        assert!(scan.stopped);
    }

    #[test]
    fn flush_releases_unmatched_tail() {
        let mut scanner = TextStopScanner::new(vec!["END".to_string()]);
        let scan = scanner.push("value EN");
        assert_eq!(scan.emit, "value ");
        assert_eq!(scanner.flush(), "EN");
        assert_eq!(scanner.matched(), None);
    }

    #[test]
    fn earliest_match_wins_across_stops() {
        let mut scanner = TextStopScanner::new(vec!["zz".to_string(), "b".to_string()]);
        let (out, stopped) = scanner.scan_complete("abzz");
        assert_eq!(out, "ab");
        assert!(stopped);
        assert_eq!(scanner.matched(), Some("b"));
    }
}
