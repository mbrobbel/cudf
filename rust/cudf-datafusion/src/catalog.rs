// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Simple catalog mapping table names to GPU tables with column metadata.

use std::collections::HashMap;

use cudf::data_type::TypeId;
use cudf::table::Table;
use datafusion::arrow::datatypes::{DataType as ArrowDataType, Field, Schema};

/// A GPU table entry in the catalog, bundling the table data with its schema.
pub struct GpuTable {
    /// The GPU-resident table.
    pub table: Table,
    /// Column names in positional order.
    pub column_names: Vec<String>,
    /// Arrow schema for DataFusion registration (derived from cudf types + names).
    pub schema: Schema,
}

/// A simple in-memory catalog mapping table names to [`GpuTable`] entries.
#[derive(Default)]
pub struct GpuCatalog {
    tables: HashMap<String, GpuTable>,
}

impl GpuCatalog {
    /// Creates an empty catalog.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a GPU table under the given name.
    ///
    /// The Arrow schema is derived from the cudf table's column types and the
    /// provided column names.
    pub fn register(
        &mut self,
        name: &str,
        table: Table,
        column_names: Vec<String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let schema = derive_schema(&table, &column_names)?;
        self.tables.insert(
            name.to_string(),
            GpuTable {
                table,
                column_names,
                schema,
            },
        );
        Ok(())
    }

    /// Looks up a table by name.
    pub fn get(&self, name: &str) -> Option<&GpuTable> {
        self.tables.get(name)
    }

    /// Returns an iterator over all (name, table) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &GpuTable)> {
        self.tables.iter()
    }
}

/// Maps a cudf `TypeId` to a DataFusion-compatible Arrow `DataType`.
pub fn type_id_to_df_arrow(tid: TypeId) -> Result<ArrowDataType, Box<dyn std::error::Error>> {
    match tid {
        TypeId::INT8 => Ok(ArrowDataType::Int8),
        TypeId::INT16 => Ok(ArrowDataType::Int16),
        TypeId::INT32 => Ok(ArrowDataType::Int32),
        TypeId::INT64 => Ok(ArrowDataType::Int64),
        TypeId::UINT8 => Ok(ArrowDataType::UInt8),
        TypeId::UINT16 => Ok(ArrowDataType::UInt16),
        TypeId::UINT32 => Ok(ArrowDataType::UInt32),
        TypeId::UINT64 => Ok(ArrowDataType::UInt64),
        TypeId::FLOAT32 => Ok(ArrowDataType::Float32),
        TypeId::FLOAT64 => Ok(ArrowDataType::Float64),
        TypeId::BOOL8 => Ok(ArrowDataType::Boolean),
        TypeId::STRING => Ok(ArrowDataType::Utf8),
        TypeId::TIMESTAMP_DAYS => Ok(ArrowDataType::Date32),
        _ => Err(format!("Unsupported cudf TypeId for Arrow conversion: {tid}").into()),
    }
}

/// Derives an Arrow schema from a cudf Table and column names.
fn derive_schema(
    table: &Table,
    column_names: &[String],
) -> Result<Schema, Box<dyn std::error::Error>> {
    let ncols = table.columns_len();
    if ncols != column_names.len() {
        return Err(format!(
            "column count mismatch: table has {ncols} columns but {} names provided",
            column_names.len()
        )
        .into());
    }

    let mut fields = Vec::with_capacity(ncols);
    for (i, col_name) in column_names.iter().enumerate() {
        let view = table.column(i)?;
        let tid = view.type_id();
        let arrow_dt = type_id_to_df_arrow(tid)?;
        fields.push(Field::new(col_name, arrow_dt, true));
    }

    Ok(Schema::new(fields))
}
