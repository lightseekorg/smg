use openmetrics_parser::{MetricFamily, MetricsExposition, PrometheusType, PrometheusValue};
use tracing::warn;

#[derive(Debug)]
pub struct MetricPack {
    pub labels: Vec<(String, String)>,
    pub metrics_text: String,
}

type PrometheusExposition = MetricsExposition<PrometheusType, PrometheusValue>;
type PrometheusFamily = MetricFamily<PrometheusType, PrometheusValue>;

pub fn aggregate_metrics(metric_packs: Vec<MetricPack>) -> anyhow::Result<String> {
    let mut expositions = vec![];
    for metric_pack in metric_packs {
        let encoded = normalize_metric_names(&metric_pack.metrics_text);

        let exposition = match openmetrics_parser::prometheus::parse_prometheus(&encoded) {
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

/// Normalize metric names so TYPE/HELP headers match sample lines.
/// Newer sglang uses OpenMetrics convention: `# TYPE sglang:foo counter`
/// with samples as `sglang_foo_total`. We convert colons to underscores
/// and align counter `_total` suffixes so the parser sees consistent names.
/// Colons inside label values are left untouched.
fn normalize_metric_names(text: &str) -> String {
    let mut sample_names: std::collections::HashSet<String> = std::collections::HashSet::new();
    for line in text.lines() {
        if !line.starts_with('#') && !line.is_empty() {
            let end = line.find(|c: char| c == '{' || c == ' ').unwrap_or(line.len());
            sample_names.insert(line[..end].replace(':', "_"));
        }
    }

    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        if line.starts_with("# HELP ") || line.starts_with("# TYPE ") {
            let rest = &line[7..];
            let name_end = rest.find(' ').unwrap_or(rest.len());
            let mut normalized = rest[..name_end].replace(':', "_");
            if !sample_names.contains(&normalized) {
                let with_total = format!("{normalized}_total");
                if sample_names.contains(&with_total) {
                    normalized = with_total;
                }
            }
            out.push_str(&line[..7]);
            out.push_str(&normalized);
            out.push_str(&rest[name_end..]);
        } else if !line.starts_with('#') && !line.is_empty() {
            let end = line.find(|c: char| c == '{' || c == ' ').unwrap_or(line.len());
            out.push_str(&line[..end].replace(':', "_"));
            out.push_str(&line[end..]);
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    out
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
    let (a, b) = align_labels(a, b);
    a.with_samples(b.into_iter_samples())
        .map_err(|e| anyhow::anyhow!("failed to merge samples: {e:?}"))
}

/// Pad missing labels with `""` so both families share the same label set.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consistent_colons_normalized_to_underscores() {
        let input = concat!(
            "# HELP sglang:num_running_reqs Number of running requests.\n",
            "# TYPE sglang:num_running_reqs gauge\n",
            "sglang:num_running_reqs 42\n",
        );
        let packs = vec![MetricPack {
            labels: vec![("worker".into(), "w0".into())],
            metrics_text: input.into(),
        }];
        let output = aggregate_metrics(packs).unwrap();
        assert!(output.contains("sglang_num_running_reqs"), "{output}");
    }

    #[test]
    fn mixed_colon_underscore_counter_convention() {
        let input = concat!(
            "# HELP sglang:cuda_graph_passes Total CUDA graph passes.\n",
            "# TYPE sglang:cuda_graph_passes counter\n",
            "sglang_cuda_graph_passes_total 7.0\n",
        );
        let packs = vec![MetricPack {
            labels: vec![],
            metrics_text: input.into(),
        }];
        let output = aggregate_metrics(packs).unwrap();
        assert!(output.contains("sglang_cuda_graph_passes_total"), "{output}");
    }

    #[test]
    fn colons_in_label_values_preserved() {
        let input = concat!(
            "# HELP up Target is up.\n",
            "# TYPE up gauge\n",
            "up{addr=\"grpc://host:9001\"} 1\n",
        );
        let packs = vec![MetricPack {
            labels: vec![],
            metrics_text: input.into(),
        }];
        let output = aggregate_metrics(packs).unwrap();
        assert!(output.contains("grpc://host:9001"), "{output}");
    }

    #[test]
    fn colonless_names_unchanged() {
        let input = concat!(
            "# HELP smg_http_requests_total Total HTTP requests.\n",
            "# TYPE smg_http_requests_total counter\n",
            "smg_http_requests_total 100\n",
        );
        let packs = vec![MetricPack {
            labels: vec![],
            metrics_text: input.into(),
        }];
        let output = aggregate_metrics(packs).unwrap();
        assert!(output.contains("smg_http_requests_total"), "{output}");
    }
}
