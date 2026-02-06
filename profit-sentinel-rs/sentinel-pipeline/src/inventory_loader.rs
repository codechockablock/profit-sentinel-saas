//! CSV inventory data loader.
//!
//! Parses inventory CSV files into `InventoryRow` structs with store context.
//! Expected CSV columns:
//!   store_id, sku, qty_on_hand, unit_cost, margin_pct, sales_last_30d,
//!   days_since_receipt, retail_price, is_damaged, on_order_qty, is_seasonal

use sentinel_vsa::bundling::InventoryRow;
use serde::Deserialize;
use std::io::Read;

/// A CSV record with store_id included.
/// The pipeline needs store_id for grouping, but InventoryRow doesn't have it.
#[derive(Debug, Clone, Deserialize)]
pub struct InventoryRecord {
    pub store_id: String,
    pub sku: String,
    pub qty_on_hand: f64,
    pub unit_cost: f64,
    pub margin_pct: f64,
    pub sales_last_30d: f64,
    pub days_since_receipt: f64,
    pub retail_price: f64,
    #[serde(deserialize_with = "deserialize_bool")]
    pub is_damaged: bool,
    pub on_order_qty: f64,
    #[serde(deserialize_with = "deserialize_bool")]
    pub is_seasonal: bool,
}

impl InventoryRecord {
    /// Convert to a VSA-compatible InventoryRow (drops store_id).
    pub fn to_inventory_row(&self) -> InventoryRow {
        InventoryRow {
            sku: self.sku.clone(),
            qty_on_hand: self.qty_on_hand,
            unit_cost: self.unit_cost,
            margin_pct: self.margin_pct,
            sales_last_30d: self.sales_last_30d,
            days_since_receipt: self.days_since_receipt,
            retail_price: self.retail_price,
            is_damaged: self.is_damaged,
            on_order_qty: self.on_order_qty,
            is_seasonal: self.is_seasonal,
        }
    }
}

/// Load inventory records from a CSV reader.
pub fn load_inventory<R: Read>(reader: R) -> Result<Vec<InventoryRecord>, String> {
    let mut csv_reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .trim(csv::Trim::All)
        .from_reader(reader);

    let mut records = Vec::new();
    for (line_num, result) in csv_reader.deserialize().enumerate() {
        let record: InventoryRecord = result
            .map_err(|e| format!("CSV parse error at line {}: {}", line_num + 2, e))?;
        records.push(record);
    }

    Ok(records)
}

/// Load inventory records from a CSV file path.
pub fn load_inventory_file(path: &str) -> Result<Vec<InventoryRecord>, String> {
    let file = std::fs::File::open(path)
        .map_err(|e| format!("Failed to open '{}': {}", path, e))?;
    load_inventory(file)
}

/// Group records by store_id.
pub fn group_by_store(records: &[InventoryRecord]) -> Vec<(String, Vec<InventoryRecord>)> {
    let mut groups: std::collections::HashMap<String, Vec<InventoryRecord>> =
        std::collections::HashMap::new();
    for record in records {
        groups
            .entry(record.store_id.clone())
            .or_default()
            .push(record.clone());
    }
    let mut result: Vec<_> = groups.into_iter().collect();
    result.sort_by(|a, b| a.0.cmp(&b.0));
    result
}

/// Flexible bool deserializer: handles "true"/"false", "1"/"0", "yes"/"no".
fn deserialize_bool<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    match s.to_lowercase().trim() {
        "true" | "1" | "yes" | "y" => Ok(true),
        "false" | "0" | "no" | "n" | "" => Ok(false),
        other => Err(serde::de::Error::custom(format!(
            "expected bool value, got '{}'",
            other
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_CSV: &str = "\
store_id,sku,qty_on_hand,unit_cost,margin_pct,sales_last_30d,days_since_receipt,retail_price,is_damaged,on_order_qty,is_seasonal
store-7,ELC-4401,-47,23.50,0.35,10,30,31.73,false,0,false
store-7,SEA-1201,100,50.00,0.35,0,180,67.50,false,0,false
store-12,PNT-1001,50,100.00,0.05,10,30,105.00,false,0,false
store-12,HRD-9901,10,200.00,0.30,5,15,260.00,true,25,false
store-3,GRD-5501,300,25.00,0.35,5,90,33.75,false,0,true
";

    #[test]
    fn load_sample_csv() {
        let records = load_inventory(SAMPLE_CSV.as_bytes()).unwrap();
        assert_eq!(records.len(), 5);
        assert_eq!(records[0].store_id, "store-7");
        assert_eq!(records[0].sku, "ELC-4401");
        assert!((records[0].qty_on_hand - (-47.0)).abs() < 0.01);
        assert!((records[0].unit_cost - 23.50).abs() < 0.01);
        assert!(!records[0].is_damaged);
        assert!(records[3].is_damaged);
        assert!(records[4].is_seasonal);
    }

    #[test]
    fn group_records_by_store() {
        let records = load_inventory(SAMPLE_CSV.as_bytes()).unwrap();
        let groups = group_by_store(&records);
        assert_eq!(groups.len(), 3); // store-3, store-7, store-12
        let store_7 = groups.iter().find(|(id, _)| id == "store-7").unwrap();
        assert_eq!(store_7.1.len(), 2);
    }

    #[test]
    fn to_inventory_row_preserves_fields() {
        let records = load_inventory(SAMPLE_CSV.as_bytes()).unwrap();
        let row = records[0].to_inventory_row();
        assert_eq!(row.sku, "ELC-4401");
        assert!((row.qty_on_hand - (-47.0)).abs() < 0.01);
        assert!((row.unit_cost - 23.50).abs() < 0.01);
    }

    #[test]
    fn bool_parsing_handles_variants() {
        let csv_data = "\
store_id,sku,qty_on_hand,unit_cost,margin_pct,sales_last_30d,days_since_receipt,retail_price,is_damaged,on_order_qty,is_seasonal
s1,A,10,10,0.3,5,30,13,1,0,0
s1,B,10,10,0.3,5,30,13,yes,0,no
s1,C,10,10,0.3,5,30,13,true,0,false
";
        let records = load_inventory(csv_data.as_bytes()).unwrap();
        assert!(records[0].is_damaged);
        assert!(records[1].is_damaged);
        assert!(records[2].is_damaged);
        assert!(!records[0].is_seasonal);
        assert!(!records[1].is_seasonal);
        assert!(!records[2].is_seasonal);
    }
}
