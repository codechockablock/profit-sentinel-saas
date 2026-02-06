use std::collections::HashMap;
use std::env;
use std::process;
use std::time::Instant;

use chrono::Utc;
use serde::Serialize;

use sentinel_pipeline::candidate_pipeline::CandidatePipeline;
use sentinel_pipeline::inventory_loader::{load_inventory_file, InventoryRecord};
use sentinel_pipeline::pipelines::executive_digest::ExecutiveDigestPipeline;
use sentinel_pipeline::types::{AgentQuery, IssueCandidate, TimeRange, UserRole};

// ---------------------------------------------------------------------------
// JSON output contract
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct DigestJson {
    generated_at: String,
    store_filter: Vec<String>,
    pipeline_ms: u128,
    issues: Vec<IssueJson>,
    summary: SummaryJson,
}

#[derive(Serialize)]
struct CauseScoreJson {
    cause: String,
    score: f64,
    evidence_count: usize,
}

#[derive(Serialize)]
struct IssueJson {
    id: String,
    issue_type: String,
    store_id: String,
    dollar_impact: f64,
    confidence: f64,
    trend_direction: String,
    priority_score: f64,
    urgency_score: f64,
    detection_timestamp: String,
    skus: Vec<SkuJson>,
    context: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    root_cause: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    root_cause_confidence: Option<f64>,
    // Phase 13: detailed evidence for symbolic bridge
    #[serde(skip_serializing_if = "Vec::is_empty")]
    cause_scores: Vec<CauseScoreJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    root_cause_ambiguity: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    active_signals: Vec<String>,
}

#[derive(Serialize)]
struct SkuJson {
    sku_id: String,
    qty_on_hand: f64,
    unit_cost: f64,
    retail_price: f64,
    margin_pct: f64,
    sales_last_30d: f64,
    days_since_receipt: f64,
    is_damaged: bool,
    on_order_qty: f64,
    is_seasonal: bool,
}

#[derive(Serialize)]
struct SummaryJson {
    total_issues: usize,
    total_dollar_impact: f64,
    stores_affected: usize,
    records_processed: usize,
    issues_detected: usize,
    issues_filtered_out: usize,
}

fn issue_type_str(candidate: &IssueCandidate) -> String {
    format!("{:?}", candidate.issue_type)
}

fn trend_str(candidate: &IssueCandidate) -> String {
    format!("{:?}", candidate.trend_direction)
}

/// Build SKU detail records by matching candidate SKU IDs back to inventory via HashMap.
fn build_sku_details(
    candidate: &IssueCandidate,
    record_index: &HashMap<(&str, &str), &InventoryRecord>,
) -> Vec<SkuJson> {
    candidate
        .sku_ids
        .iter()
        .filter_map(|sku_id| {
            record_index
                .get(&(candidate.store_id.as_str(), sku_id.as_str()))
                .map(|r| SkuJson {
                    sku_id: r.sku.clone(),
                    qty_on_hand: r.qty_on_hand,
                    unit_cost: r.unit_cost,
                    retail_price: r.retail_price,
                    margin_pct: r.margin_pct,
                    sales_last_30d: r.sales_last_30d,
                    days_since_receipt: r.days_since_receipt,
                    is_damaged: r.is_damaged,
                    on_order_qty: r.on_order_qty,
                    is_seasonal: r.is_seasonal,
                })
        })
        .collect()
}

/// Look up a record by (store_id, sku) from the pre-built index.
fn lookup_record<'a>(
    record_index: &'a HashMap<(&str, &str), &'a InventoryRecord>,
    store_id: &str,
    sku: &str,
) -> Option<&'a InventoryRecord> {
    record_index.get(&(store_id, sku)).copied()
}

/// Generate a human-readable context string for an issue.
fn generate_context(
    candidate: &IssueCandidate,
    record_index: &HashMap<(&str, &str), &InventoryRecord>,
    store_record_count: &HashMap<&str, usize>,
) -> String {
    let sku_count = candidate.sku_ids.len();
    let store_id = candidate.store_id.as_str();
    let total_store_skus = store_record_count
        .get(store_id)
        .copied()
        .unwrap_or(0);

    match format!("{:?}", candidate.issue_type).as_str() {
        "NegativeInventory" => {
            if sku_count == 1 {
                let sku = &candidate.sku_ids[0];
                if let Some(r) = lookup_record(record_index, store_id, sku) {
                    format!(
                        "{} units short at ${:.2}/unit. System shows negative on-hand requiring investigation.",
                        r.qty_on_hand.abs() as i64,
                        r.unit_cost
                    )
                } else {
                    "Negative on-hand quantity detected. Investigate possible shrinkage or system error.".into()
                }
            } else {
                format!(
                    "{} SKUs with negative on-hand in this store. Potential systemic receiving or shrinkage issue.",
                    sku_count
                )
            }
        }
        "DeadStock" => {
            let avg_days: f64 = candidate.sku_ids.iter()
                .filter_map(|sku| lookup_record(record_index, store_id, sku))
                .map(|r| r.days_since_receipt)
                .sum::<f64>()
                / sku_count.max(1) as f64;
            format!(
                "{} SKU{} with zero sales for {:.0}+ days. Capital tied up in non-moving inventory.",
                sku_count,
                if sku_count > 1 { "s" } else { "" },
                avg_days
            )
        }
        "MarginErosion" => {
            let avg_margin: f64 = candidate.sku_ids.iter()
                .filter_map(|sku| lookup_record(record_index, store_id, sku))
                .map(|r| r.margin_pct)
                .sum::<f64>()
                / sku_count.max(1) as f64;
            format!(
                "{} SKU{} averaging {:.0}% margin vs 35% benchmark. Pricing review recommended.",
                sku_count,
                if sku_count > 1 { "s" } else { "" },
                avg_margin * 100.0
            )
        }
        "VendorShortShip" => {
            let has_on_order = candidate.sku_ids.iter().any(|sku| {
                lookup_record(record_index, store_id, sku)
                    .map(|r| r.on_order_qty > 0.0)
                    .unwrap_or(false)
            });
            if has_on_order {
                format!(
                    "Damaged goods with active purchase orders. Vendor fulfillment issue — contact vendor.",
                )
            } else {
                "Damaged inventory detected. Evaluate vendor quality and file claim if applicable.".into()
            }
        }
        "PatronageMiss" => {
            format!(
                "Seasonal overstock: {} SKU{} past sales window. Markdown or return to reduce carrying costs.",
                sku_count,
                if sku_count > 1 { "s" } else { "" },
            )
        }
        "PurchasingLeakage" => {
            format!(
                "High-cost items at below-benchmark margins. Review vendor pricing and negotiate volume discounts.",
            )
        }
        "ReceivingGap" => {
            format!(
                "Pricing anomaly: {} SKU{} with negative retail price. Data correction needed.",
                sku_count,
                if sku_count > 1 { "s" } else { "" },
            )
        }
        "ShrinkagePattern" => {
            let total_value: f64 = candidate.sku_ids.iter()
                .filter_map(|sku| lookup_record(record_index, store_id, sku))
                .map(|r| r.qty_on_hand * r.unit_cost)
                .sum();
            format!(
                "{} SKU{} with high inventory value (${:.0}) but near-zero margin and minimal sales. Investigate potential shrinkage.",
                sku_count,
                if sku_count > 1 { "s" } else { "" },
                total_value,
            )
        }
        "ZeroCostAnomaly" => {
            let has_sales = candidate.sku_ids.iter().any(|sku| {
                lookup_record(record_index, store_id, sku)
                    .map(|r| r.sales_last_30d > 0.0)
                    .unwrap_or(false)
            });
            if has_sales {
                format!(
                    "{} SKU{} with $0 cost and active sales. Margin calculations are invalid — update cost data urgently.",
                    sku_count,
                    if sku_count > 1 { "s" } else { "" },
                )
            } else {
                format!(
                    "{} SKU{} with $0 cost in system. Missing cost data corrupts profitability analysis.",
                    sku_count,
                    if sku_count > 1 { "s" } else { "" },
                )
            }
        }
        "PriceDiscrepancy" => {
            let avg_loss: f64 = candidate.sku_ids.iter()
                .filter_map(|sku| lookup_record(record_index, store_id, sku))
                .map(|r| if r.unit_cost > r.retail_price { r.unit_cost - r.retail_price } else { 0.0 })
                .sum::<f64>()
                / sku_count.max(1) as f64;
            format!(
                "{} SKU{} priced below cost (avg ${:.2}/unit loss). Correct retail pricing or review vendor costs.",
                sku_count,
                if sku_count > 1 { "s" } else { "" },
                avg_loss,
            )
        }
        "Overstock" => {
            let avg_months: f64 = candidate.sku_ids.iter()
                .filter_map(|sku| lookup_record(record_index, store_id, sku))
                .map(|r| if r.sales_last_30d > 0.0 { r.qty_on_hand / r.sales_last_30d } else { 24.0 })
                .sum::<f64>()
                / sku_count.max(1) as f64;
            format!(
                "{} SKU{} with {:.0}+ months supply on hand. Reduce orders or arrange transfers to reduce carrying cost.",
                sku_count,
                if sku_count > 1 { "s" } else { "" },
                avg_months,
            )
        }
        _ => format!(
            "{} SKUs affected across {} total store records.",
            sku_count, total_store_skus
        ),
    }
}

fn build_json(
    result: &sentinel_pipeline::candidate_pipeline::PipelineResult<AgentQuery, IssueCandidate>,
    records: &[InventoryRecord],
    store_filter: &[String],
    total_records: usize,
    pipeline_ms: u128,
) -> DigestJson {
    // Pre-build HashMap for O(1) record lookups instead of O(N) linear searches.
    // Key: (store_id, sku) → &InventoryRecord
    let record_index: HashMap<(&str, &str), &InventoryRecord> = records
        .iter()
        .map(|r| ((r.store_id.as_str(), r.sku.as_str()), r))
        .collect();

    // Pre-count records per store for context generation.
    let mut store_record_count: HashMap<&str, usize> = HashMap::new();
    for r in records {
        *store_record_count.entry(r.store_id.as_str()).or_insert(0) += 1;
    }

    let mut stores_affected: Vec<String> = result
        .selected_candidates
        .iter()
        .map(|c| c.store_id.clone())
        .collect();
    stores_affected.sort();
    stores_affected.dedup();

    let total_impact: f64 = result.selected_candidates.iter().map(|c| c.dollar_impact).sum();

    DigestJson {
        generated_at: Utc::now().to_rfc3339(),
        store_filter: store_filter.to_vec(),
        pipeline_ms,
        issues: result
            .selected_candidates
            .iter()
            .map(|c| IssueJson {
                id: c.id.clone(),
                issue_type: issue_type_str(c),
                store_id: c.store_id.clone(),
                dollar_impact: c.dollar_impact,
                confidence: c.confidence,
                trend_direction: trend_str(c),
                priority_score: c.priority_score.unwrap_or(0.0),
                urgency_score: c.urgency_score.unwrap_or(0.0),
                detection_timestamp: c.detection_timestamp.clone(),
                skus: build_sku_details(c, &record_index),
                context: generate_context(c, &record_index, &store_record_count),
                root_cause: c.root_cause.as_ref().map(|rc| format!("{:?}", rc)),
                root_cause_confidence: c.root_cause_confidence,
                cause_scores: c.cause_scores.iter().map(|cs| CauseScoreJson {
                    cause: format!("{:?}", cs.cause),
                    score: cs.score,
                    evidence_count: cs.evidence_count,
                }).collect(),
                root_cause_ambiguity: c.root_cause_ambiguity,
                active_signals: c.active_signals.clone(),
            })
            .collect(),
        summary: SummaryJson {
            total_issues: result.selected_candidates.len(),
            total_dollar_impact: total_impact,
            stores_affected: stores_affected.len(),
            records_processed: total_records,
            issues_detected: result.retrieved_candidates.len(),
            issues_filtered_out: result.filtered_candidates.len(),
        },
    }
}

// ---------------------------------------------------------------------------
// Human-readable output
// ---------------------------------------------------------------------------

/// Format a number with comma thousands separators.
fn format_dollars(amount: f64) -> String {
    let whole = amount.abs() as u64;
    let sign = if amount < 0.0 { "-" } else { "" };

    if whole < 1_000 {
        return format!("{}{}", sign, whole);
    }

    let s = whole.to_string();
    let mut result = String::new();
    for (i, ch) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(ch);
    }
    format!("{}{}", sign, result.chars().rev().collect::<String>())
}

fn print_human(
    result: &sentinel_pipeline::candidate_pipeline::PipelineResult<AgentQuery, IssueCandidate>,
    query_store_ids: &[String],
    total_records: usize,
    load_ms: u128,
    pipeline_ms: u128,
) {
    println!();
    println!("  \u{2554}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2557}");
    println!("  \u{2551}          PROFIT SENTINEL \u{2014} Executive Morning Digest         \u{2551}");
    println!("  \u{255a}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{255d}");
    println!();

    let total_impact: f64 = result
        .selected_candidates
        .iter()
        .map(|c| c.dollar_impact)
        .sum();
    let kept_count = result.retrieved_candidates.len() - result.filtered_candidates.len();
    println!(
        "  {} stores analyzed  \u{00b7}  {} records processed  \u{00b7}  {} issues detected",
        query_store_ids.len(),
        total_records,
        result.retrieved_candidates.len()
    );
    println!(
        "  {} passed filters ({} removed)  \u{00b7}  Top {} selected  \u{00b7}  ${} total exposure",
        kept_count,
        result.filtered_candidates.len(),
        result.selected_candidates.len(),
        format_dollars(total_impact)
    );
    println!();

    if result.selected_candidates.is_empty() {
        println!("  No actionable issues detected. All clear!");
    } else {
        println!("  {:\u{2500}<64}", "");
        for (i, c) in result.selected_candidates.iter().enumerate() {
            let priority = c.priority_score.unwrap_or(0.0);
            let urgency_icon = match priority {
                p if p >= 8.0 => "!!",
                p if p >= 6.0 => "! ",
                _ => "  ",
            };

            let impact_str = format!("${}", format_dollars(c.dollar_impact));

            println!(
                "  {} {}. {:10} {:20} {:>10}  score {:.1} {}",
                urgency_icon,
                i + 1,
                c.store_id,
                format!("{}", c.issue_type),
                impact_str,
                priority,
                c.trend_direction,
            );

            let sku_display = if c.sku_ids.len() <= 4 {
                c.sku_ids.join(", ")
            } else {
                format!(
                    "{}, +{} more",
                    c.sku_ids[..3].join(", "),
                    c.sku_ids.len() - 3
                )
            };
            let cause_display = if let Some(ref rc) = c.root_cause {
                let rc_conf = c.root_cause_confidence.unwrap_or(0.0);
                format!("  cause: {} ({:.0}%)", rc, rc_conf * 100.0)
            } else {
                String::new()
            };
            println!(
                "       SKUs: {}  (confidence {:.0}%){}",
                sku_display,
                c.confidence * 100.0,
                cause_display,
            );
            println!();
        }
        println!("  {:\u{2500}<64}", "");
    }

    println!();
    println!(
        "  \u{23f1}  CSV loaded in {}ms \u{00b7} Pipeline ran in {}ms \u{00b7} Total {}ms",
        load_ms,
        pipeline_ms,
        load_ms + pipeline_ms
    );
    println!();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: sentinel-server <inventory.csv> [--stores s1,s2,...] [--top N] [--json]");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --stores   Comma-separated store IDs to analyze");
        eprintln!("  --top      Number of top issues to return (default: 5)");
        eprintln!("  --json     Output as JSON instead of formatted text");
        eprintln!();
        eprintln!("Example:");
        eprintln!("  sentinel-server fixtures/sample_inventory.csv");
        eprintln!("  sentinel-server fixtures/sample_inventory.csv --json");
        eprintln!("  sentinel-server fixtures/sample_inventory.csv --stores store-7,store-12 --top 3 --json");
        process::exit(1);
    }

    let csv_path = &args[1];

    // Parse optional flags
    let mut store_filter: Option<Vec<String>> = None;
    let mut top_k: usize = 5;
    let mut json_output = false;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--stores" => {
                if i + 1 < args.len() {
                    store_filter = Some(
                        args[i + 1]
                            .split(',')
                            .map(|s| s.trim().to_string())
                            .collect(),
                    );
                    i += 2;
                } else {
                    eprintln!("Error: --stores requires a comma-separated list of store IDs");
                    process::exit(1);
                }
            }
            "--top" => {
                if i + 1 < args.len() {
                    top_k = args[i + 1].parse().unwrap_or_else(|_| {
                        eprintln!("Error: --top requires a positive integer");
                        process::exit(1);
                    });
                    i += 2;
                } else {
                    eprintln!("Error: --top requires a number");
                    process::exit(1);
                }
            }
            "--json" => {
                json_output = true;
                i += 1;
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                process::exit(1);
            }
        }
    }

    // Load inventory data from CSV
    let load_start = Instant::now();
    let records = match load_inventory_file(csv_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error loading CSV: {}", e);
            process::exit(1);
        }
    };
    let load_ms = load_start.elapsed().as_millis();

    let total_records = records.len();

    // Discover unique store IDs
    let mut store_ids: Vec<String> = records.iter().map(|r| r.store_id.clone()).collect();
    store_ids.sort();
    store_ids.dedup();

    // Apply store filter if provided
    let query_store_ids: Vec<String> = if let Some(ref filter) = store_filter {
        store_ids
            .iter()
            .filter(|s| filter.contains(s))
            .cloned()
            .collect()
    } else {
        store_ids.clone()
    };

    if query_store_ids.is_empty() {
        eprintln!("Error: no matching stores found in the data");
        if let Some(ref filter) = store_filter {
            eprintln!("  Requested: {:?}", filter);
        }
        eprintln!("  Available: {:?}", store_ids);
        process::exit(1);
    }

    // Build and run pipeline
    let pipeline_start = Instant::now();
    let pipeline = ExecutiveDigestPipeline::with_inventory_and_size(records.clone(), top_k);

    let query = AgentQuery {
        request_id: "digest-001".into(),
        user_id: "exec_001".into(),
        user_role: UserRole::Executive,
        store_ids: query_store_ids.clone(),
        time_range: TimeRange {
            start: String::new(),
            end: String::new(),
        },
        priority_filters: None,
    };

    let result = pipeline.execute(query).await;
    let pipeline_ms = pipeline_start.elapsed().as_millis();

    if json_output {
        let digest = build_json(&result, &records, &query_store_ids, total_records, pipeline_ms);
        println!("{}", serde_json::to_string_pretty(&digest).unwrap());
    } else {
        print_human(&result, &query_store_ids, total_records, load_ms, pipeline_ms);
    }
}
