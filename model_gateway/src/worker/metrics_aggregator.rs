use openmetrics_parser::{MetricFamily, MetricsExposition, PrometheusType, PrometheusValue};
use tracing::warn;

#[derive(Debug)]
pub struct MetricPack {
    pub labels: Vec<(String, String)>,
    pub metrics_text: String,
}

type PrometheusExposition = MetricsExposition<PrometheusType, PrometheusValue>;
type PrometheusFamily = MetricFamily<PrometheusType, PrometheusValue>;

/// Aggregate Prometheus metrics scraped from multiple sources into a unified one
pub fn aggregate_metrics(metric_packs: Vec<MetricPack>) -> anyhow::Result<String> {
    let mut expositions = vec![];
    for metric_pack in metric_packs {
        let metrics_text = normalize_metrics_text(&metric_pack.metrics_text);

        let exposition = match openmetrics_parser::prometheus::parse_prometheus(&metrics_text) {
            Ok(x) => x,
            Err(err) => {
                warn!(
                    "aggregate_metrics error when parsing text: pack={:?} err={:?}",
                    metric_pack, err
                );
                continue;
            }
        };
        let exposition = transform_metrics(exposition, &metric_pack.labels);
        expositions.push(exposition);
    }

    let text = try_reduce(expositions, merge_exposition)?
        .map(|x| format!("{x}"))
        .unwrap_or_default();
    Ok(text)
}

/// Normalize Prometheus metrics text for parsing.
///
/// Handles the format emitted by sglang (prometheus_client 0.25+) where
/// HELP/TYPE lines use colons and omit suffixes (e.g. `sglang:prompt_tokens`)
/// while data lines use underscores with standard Prometheus suffixes
/// (e.g. `sglang_prompt_tokens_total`).  The `openmetrics_parser` Prometheus
/// parser requires HELP/TYPE names to include the suffix for counters
/// (`_total`) and to match data line base names for histograms.
///
/// Also strips OpenMetrics `# EOF` markers.
fn normalize_metrics_text(text: &str) -> String {
    // Two-pass approach:
    //   Pass 1 — collect (descriptor_name → first_sample_name) mapping
    //   Pass 2 — rewrite HELP/TYPE names and normalize colons

    let lines: Vec<&str> = text.lines().collect();

    // Pass 1: build descriptor→sample mapping
    let mut desc_to_sample: Vec<(String, Option<String>)> = Vec::new();
    let mut current_desc: Option<String> = None;

    for line in &lines {
        if line.starts_with("# HELP ") || line.starts_with("# TYPE ") {
            let rest = &line[7..];
            let name_end = rest.find([' ', '\t']).unwrap_or(rest.len());
            let raw = rest[..name_end].replace(':', "_");

            if current_desc.as_deref() != Some(&raw) {
                if let Some(prev) = current_desc.take() {
                    if !desc_to_sample.iter().any(|(d, _)| d == &prev) {
                        desc_to_sample.push((prev, None));
                    }
                }
                current_desc = Some(raw);
            }
        } else if !line.is_empty() && !line.starts_with('#') {
            let name_end = line.find(['{', ' ']).unwrap_or(line.len());
            let sample = line[..name_end].replace(':', "_");

            if let Some(ref desc) = current_desc {
                if !desc_to_sample.iter().any(|(d, _)| d == desc) {
                    desc_to_sample.push((desc.clone(), Some(sample)));
                }
            }
        }
    }
    if let Some(prev) = current_desc {
        if !desc_to_sample.iter().any(|(d, _)| d == &prev) {
            desc_to_sample.push((prev, None));
        }
    }

    // Build name rewrite map: descriptor_normalized → expected_help_name
    let mut rewrite: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    for (desc, sample) in &desc_to_sample {
        if let Some(sample_name) = sample {
            let expected = help_name_for_sample(desc, sample_name);
            if &expected != desc {
                rewrite.insert(desc.clone(), expected);
            }
        }
    }

    // Pass 2: rewrite
    let mut result = String::with_capacity(text.len());
    for line in &lines {
        if line.starts_with("# EOF") {
            continue;
        }

        if line.starts_with("# HELP ") || line.starts_with("# TYPE ") {
            let rest = &line[7..];
            let name_end = rest.find([' ', '\t']).unwrap_or(rest.len());
            let normalized = rest[..name_end].replace(':', "_");
            let final_name = rewrite.get(&normalized).unwrap_or(&normalized);
            result.push_str(&line[..7]);
            result.push_str(final_name);
            result.push_str(&rest[name_end..]);
            result.push('\n');
            continue;
        }

        if !line.is_empty() && !line.starts_with('#') {
            let name_end = line.find(['{', ' ']).unwrap_or(line.len());
            let sample_name = line[..name_end].replace(':', "_");
            result.push_str(&sample_name);
            result.push_str(&line[name_end..]);
            result.push('\n');
            continue;
        }

        result.push_str(line);
        result.push('\n');
    }

    result
}

/// Determine the correct HELP/TYPE name given a descriptor name and the first
/// sample name in that family.
///
/// For counters (`_total` suffix on sample), the HELP name must include `_total`.
/// For histograms (`_bucket`/`_sum`/`_count`), the HELP name is the base.
/// For info (`_info` suffix), the HELP name must include `_info`.
fn help_name_for_sample(desc_normalized: &str, sample_name: &str) -> String {
    // If the sample name is the descriptor name plus a known suffix, the
    // descriptor needs to be promoted to include that suffix (counter/info case).
    for suffix in &["_total", "_info", "_created"] {
        if let Some(base) = sample_name.strip_suffix(suffix) {
            if base == desc_normalized {
                return sample_name.to_string();
            }
        }
    }

    // For histogram families, the descriptor should be the base name
    // (without _bucket/_sum/_count). This is already the standard format.
    for suffix in &["_bucket", "_count", "_sum"] {
        if let Some(base) = sample_name.strip_suffix(suffix) {
            if base == desc_normalized {
                return desc_normalized.to_string();
            }
        }
    }

    // No suffix mismatch — keep the descriptor as-is
    desc_normalized.to_string()
}

fn transform_metrics(
    mut exposition: PrometheusExposition,
    extra_labels: &[(String, String)],
) -> PrometheusExposition {
    for family in exposition.families.values_mut() {
        *family = family.with_labels(extra_labels.iter().map(|(k, v)| (k.as_str(), v.as_str())));
    }
    exposition
}

fn merge_exposition(
    a: PrometheusExposition,
    b: PrometheusExposition,
) -> anyhow::Result<PrometheusExposition> {
    let mut ans = a;
    for (name, family_b) in b.families {
        let family_merged = if let Some(family_a) = ans.families.remove(&name) {
            merge_family(family_a, family_b)?
        } else {
            family_b
        };
        ans.families.insert(name, family_merged);
    }
    Ok(ans)
}

fn merge_family(a: PrometheusFamily, b: PrometheusFamily) -> anyhow::Result<PrometheusFamily> {
    // When label schemas differ (e.g., prefill vs decode workers with different DP
    // configs), pad missing labels with empty strings so both families share the
    // same label set before merging.
    let (a, b) = align_labels(a, b);
    a.with_samples(b.into_iter_samples())
        .map_err(|e| anyhow::anyhow!("failed to merge samples: {e:?}"))
}

/// Ensure two families have identical label sets by padding missing labels with `""`.
/// Returns both families unchanged if labels already match.
fn align_labels(a: PrometheusFamily, b: PrometheusFamily) -> (PrometheusFamily, PrometheusFamily) {
    let a_names = a.get_label_names();
    let b_names = b.get_label_names();
    if a_names == b_names {
        return (a, b);
    }

    let pad = |family: PrometheusFamily, other_names: &[String]| -> PrometheusFamily {
        let own_names = family.get_label_names();
        let missing: Vec<(&str, &str)> = other_names
            .iter()
            .filter(|n| !own_names.contains(n))
            .map(|n| (n.as_str(), ""))
            .collect();
        if missing.is_empty() {
            family
        } else {
            family.with_labels(missing)
        }
    };

    // Clone names before moving families into pad()
    let a_names = a_names.to_vec();
    let b_names = b_names.to_vec();
    (pad(a, &b_names), pad(b, &a_names))
}

fn try_reduce<I, T, E, F>(iterable: I, f: F) -> Result<Option<T>, E>
where
    I: IntoIterator<Item = T>,
    F: FnMut(T, T) -> Result<T, E>,
{
    let mut it = iterable.into_iter();
    let first = match it.next() {
        None => return Ok(None),
        Some(x) => x,
    };

    Ok(Some(it.try_fold(first, f)?))
}
