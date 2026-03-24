// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#[cxx::bridge(namespace = "cudf_sys")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cudf-sys/lib.hpp");

        type Table = crate::ffi::Table;
        type Column = crate::ffi::Column;

        // -- ExpressionTree --

        type ExpressionTree;

        fn new_expression_tree() -> UniquePtr<ExpressionTree>;
        fn expression_tree_len(tree: &ExpressionTree) -> usize;

        fn expression_tree_add_literal_i32(tree: Pin<&mut ExpressionTree>, value: i32) -> usize;
        fn expression_tree_add_literal_i64(tree: Pin<&mut ExpressionTree>, value: i64) -> usize;
        fn expression_tree_add_literal_f32(tree: Pin<&mut ExpressionTree>, value: f32) -> usize;
        fn expression_tree_add_literal_f64(tree: Pin<&mut ExpressionTree>, value: f64) -> usize;
        fn expression_tree_add_literal_bool(tree: Pin<&mut ExpressionTree>, value: bool) -> usize;
        fn expression_tree_add_literal_string(tree: Pin<&mut ExpressionTree>, value: &str)
        -> usize;
        fn expression_tree_add_column_ref(
            tree: Pin<&mut ExpressionTree>,
            column_index: i32,
            table_source: i32,
        ) -> usize;
        fn expression_tree_add_column_name_ref(tree: Pin<&mut ExpressionTree>, name: &str)
        -> usize;
        fn expression_tree_add_unary_op(
            tree: Pin<&mut ExpressionTree>,
            op: i32,
            operand: usize,
        ) -> usize;
        fn expression_tree_add_binary_op(
            tree: Pin<&mut ExpressionTree>,
            op: i32,
            left: usize,
            right: usize,
        ) -> usize;

        // -- Operations using ExpressionTree --

        fn ast_compute_column(
            tbl: &Table,
            tree: &ExpressionTree,
            root_index: usize,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        fn read_parquet_filtered(
            filepath: &str,
            tree: &ExpressionTree,
            root_index: usize,
        ) -> Result<UniquePtr<Table>>;

        // -- Conditional joins --

        fn conditional_inner_join(
            left: &Table,
            right: &Table,
            tree: &ExpressionTree,
            root_index: usize,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        fn conditional_left_join(
            left: &Table,
            right: &Table,
            tree: &ExpressionTree,
            root_index: usize,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        fn conditional_full_join(
            left: &Table,
            right: &Table,
            tree: &ExpressionTree,
            root_index: usize,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        fn conditional_left_semi_join(
            left: &Table,
            right: &Table,
            tree: &ExpressionTree,
            root_index: usize,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        fn conditional_left_anti_join(
            left: &Table,
            right: &Table,
            tree: &ExpressionTree,
            root_index: usize,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;
    }
}
