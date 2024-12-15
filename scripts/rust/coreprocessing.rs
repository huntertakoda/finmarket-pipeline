use csv::{ReaderBuilder, WriterBuilder};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::error::Error;
use std::time::Instant;

#[derive(Debug, Deserialize, Serialize)]
struct FinancialRecord {
    timestamp: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let start_time = Instant::now();

    let file_path = "full_financial_data.csv";
    let output_path = "processed_financial_data.csv";

    // data ingestion
    let file = File::open(file_path)?;
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);

    let records: Vec<FinancialRecord> = reader
        .deserialize()
        .filter_map(|result| result.ok())
        .collect();

    println!("Data ingestion completed. Processed {} records.", records.len());

    // data transformation, filter and process

    let processed_data: Vec<FinancialRecord> = records
        .into_par_iter()
        .filter(|record| record.volume > 1000.0)
        .map(|record| FinancialRecord {
            timestamp: record.timestamp,
            open: record.open * 1.01,
            high: record.high,
            low: record.low,
            close: record.close,
            volume: record.volume,
        })
        .collect();

    println!("Data transformation completed. Filtered down to {} records.", processed_data.len());

    // saving

    let output_file = File::create(output_path)?;
    let mut writer = WriterBuilder::new()
        .has_headers(true)
        .from_writer(output_file);

    for record in &processed_data {
        writer.serialize(record)?;
    }

    writer.flush()?;

    let duration = start_time.elapsed();
    println!("Data processing complete in {:.2?} seconds. Results saved to {}.", duration, output_path);

    Ok(())
}
