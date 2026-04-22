use openmetrics_parser::{MetricFamily, MetricsExposition, PrometheusType, PrometheusValue};
use tracing::warn;

#[derive(Debug)]
pub struct MetricPack {
    pub labels: Vec<(String, String)>,
    pub metrics_text: String,
}

type PrometheusExposition = MetricsExposition<PrometheusType, PrometheusValue>;
type PrometheusFamily = MetricFamily<PrometheusType, PrometheusValue>;

// SAFETY: openmetrics_parser rejects colons in metric names. We encode colons to
// this placeholder before parsing and decode back after serialization, preserving
// the original `namespace:metric_name` colon format in the output.
const COLON_SENTINEL: &str = "__smgcolon48f__";

/// Aggregate Prometheus metrics scraped from multiple sources into a unified one
pub fn aggregate_metrics(metric_packs: Vec<MetricPack>) -> anyhow::Result<String> {
    let mut expositions = vec![];
    for metric_pack in metric_packs {
        let metrics_text = &metric_pack.metrics_text;
        let encoded = metrics_text.replace(':', COLON_SENTINEL);

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

    let text = try_reduce(expositions.into_iter(), merge_exposition)?
        .map(|x| format!("{x}").replace(COLON_SENTINEL, ":"))
        .unwrap_or_default();
    Ok(text)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_preserves_colons_in_metric_names() {
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
        assert!(
            output.contains("sglang:num_running_reqs"),
            "colon in metric name was lost: {output}"
        );
    }

    #[test]
    fn aggregate_preserves_colons_in_label_values() {
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
        assert!(
            output.contains("grpc://host:9001"),
            "colon in label value was lost: {output}"
        );
    }

    #[test]
    fn aggregate_leaves_colonless_names_unchanged() {
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
        assert!(
            output.contains("smg_http_requests_total"),
            "colonless metric name was mangled: {output}"
        );
    }
}
