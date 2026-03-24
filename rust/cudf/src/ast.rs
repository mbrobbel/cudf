// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! AST expression trees for GPU-accelerated row-level operations.
//!
//! Expression trees enable predicate pushdown, computed columns, conditional
//! joins, and row-level filtering — operations that evaluate expressions
//! per-row on the GPU.
//!
//! # Building expressions
//!
//! Create an [`ExpressionTree`], add nodes with methods like [`col`](ExpressionTree::col),
//! [`lit_i32`](ExpressionTree::lit_i32), and [`gt`](ExpressionTree::gt), then pass the
//! tree to operations on [`Table`](crate::table::Table).
//!
//! Each method returns an [`ExprRef`] — a lightweight index into the tree.
//!
//! # Examples
//!
//! ```ignore
//! use cudf::ast::{ExpressionTree, AstOp};
//! use cudf::stream::GpuOp;
//!
//! // Compute: col(0) + col(1)
//! let mut tree = ExpressionTree::new();
//! let a = tree.col(0);
//! let b = tree.col(1);
//! let sum = tree.add(a, b);
//! let result = table.compute_column(&tree, sum).call()?;
//! # Ok::<(), cudf::error::Error>(())
//! ```

/// A lightweight, copyable index into an [`ExpressionTree`].
///
/// `ExprRef` values are returned by tree-building methods like
/// [`col`](ExpressionTree::col), [`lit_i32`](ExpressionTree::lit_i32), and
/// [`add`](ExpressionTree::add). They are used as arguments to subsequent
/// tree methods and to operations like
/// [`compute_column`](crate::table::Table::compute_column).
///
/// An `ExprRef` is only valid for the tree that created it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExprRef(usize);

impl ExprRef {
    /// Returns the raw index value.
    ///
    /// This is the zero-based position of the expression node within its
    /// [`ExpressionTree`].
    pub fn index(self) -> usize {
        self.0
    }
}

/// Identifies which table a column reference belongs to in a two-table
/// context such as a conditional join.
///
/// When building expression trees for conditional joins, column references
/// must specify which side of the join they refer to. For single-table
/// operations like [`compute_column`](crate::table::Table::compute_column),
/// the default [`Left`](TableSide::Left) side is used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(alias = "table_reference")]
pub enum TableSide {
    /// Column in the left table (default).
    Left,
    /// Column in the right table.
    Right,
}

impl TableSide {
    /// Converts to the C++ `table_reference` integer representation.
    fn to_i32(self) -> i32 {
        match self {
            TableSide::Left => 0,
            TableSide::Right => 1,
        }
    }
}

/// AST operators for unary and binary expression nodes.
///
/// This enum mirrors the C++ `cudf::ast::ast_operator` enumeration.
/// Operators are used with [`ExpressionTree::unary_op`] and
/// [`ExpressionTree::binary_op`], or through convenience methods like
/// [`add`](ExpressionTree::add) and [`gt`](ExpressionTree::gt).
///
/// # Operator categories
///
/// - **Arithmetic:** [`Add`](AstOp::Add), [`Sub`](AstOp::Sub),
///   [`Mul`](AstOp::Mul), [`Div`](AstOp::Div), [`Mod`](AstOp::Mod), etc.
/// - **Comparison:** [`Equal`](AstOp::Equal), [`Less`](AstOp::Less),
///   [`Greater`](AstOp::Greater), etc.
/// - **Logical:** [`LogicalAnd`](AstOp::LogicalAnd),
///   [`LogicalOr`](AstOp::LogicalOr), [`Not`](AstOp::Not).
/// - **Bitwise:** [`BitwiseAnd`](AstOp::BitwiseAnd),
///   [`BitwiseOr`](AstOp::BitwiseOr), [`BitwiseXor`](AstOp::BitwiseXor).
/// - **Unary math:** [`Sin`](AstOp::Sin), [`Cos`](AstOp::Cos),
///   [`Exp`](AstOp::Exp), [`Log`](AstOp::Log), [`Sqrt`](AstOp::Sqrt), etc.
/// - **Casts:** [`CastToInt64`](AstOp::CastToInt64),
///   [`CastToUint64`](AstOp::CastToUint64),
///   [`CastToFloat64`](AstOp::CastToFloat64).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(alias = "ast_operator")]
#[non_exhaustive]
pub enum AstOp {
    /// Addition (`+`).
    Add,
    /// Subtraction (`-`).
    Sub,
    /// Multiplication (`*`).
    Mul,
    /// Integer division (`/`).
    Div,
    /// True (float) division.
    TrueDiv,
    /// Floor division.
    FloorDiv,
    /// Modulus (`%`).
    Mod,
    /// Python-style modulus (always non-negative for positive divisor).
    PyMod,
    /// Exponentiation (`**`).
    Pow,
    /// Equality (`==`).
    Equal,
    /// Null-equal: treats two nulls as equal.
    NullEqual,
    /// Not-equal (`!=`).
    NotEqual,
    /// Less than (`<`).
    Less,
    /// Greater than (`>`).
    Greater,
    /// Less than or equal (`<=`).
    LessEqual,
    /// Greater than or equal (`>=`).
    GreaterEqual,
    /// Bitwise AND (`&`).
    BitwiseAnd,
    /// Bitwise OR (`|`).
    BitwiseOr,
    /// Bitwise XOR (`^`).
    BitwiseXor,
    /// Logical AND (`&&`).
    LogicalAnd,
    /// Null-logical AND: null && true => null.
    NullLogicalAnd,
    /// Logical OR (`||`).
    LogicalOr,
    /// Null-logical OR: null || false => null.
    NullLogicalOr,
    /// Identity (unary, returns the operand unchanged).
    Identity,
    /// Returns true if the operand is null (unary).
    IsNull,
    /// Sine (unary).
    Sin,
    /// Cosine (unary).
    Cos,
    /// Tangent (unary).
    Tan,
    /// Arcsine (unary).
    Arcsin,
    /// Arccosine (unary).
    Arccos,
    /// Arctangent (unary).
    Arctan,
    /// Hyperbolic sine (unary).
    Sinh,
    /// Hyperbolic cosine (unary).
    Cosh,
    /// Hyperbolic tangent (unary).
    Tanh,
    /// Inverse hyperbolic sine (unary).
    Arcsinh,
    /// Inverse hyperbolic cosine (unary).
    Arccosh,
    /// Inverse hyperbolic tangent (unary).
    Arctanh,
    /// Exponential e^x (unary).
    Exp,
    /// Natural logarithm (unary).
    Log,
    /// Square root (unary).
    Sqrt,
    /// Cube root (unary).
    Cbrt,
    /// Ceiling (unary).
    Ceil,
    /// Floor (unary).
    Floor,
    /// Absolute value (unary).
    Abs,
    /// Round to nearest integer (unary).
    Rint,
    /// Bitwise inversion / complement (unary).
    BitInvert,
    /// Logical NOT (unary).
    Not,
    /// Cast to `INT64`.
    CastToInt64,
    /// Cast to `UINT64`.
    CastToUint64,
    /// Cast to `FLOAT64`.
    CastToFloat64,
}

impl AstOp {
    /// Converts this operator to its C++ `ast_operator` integer value.
    ///
    /// The mapping is sequential: `Add`=0, `Sub`=1, ..., `CastToFloat64`=49.
    pub fn to_i32(self) -> i32 {
        match self {
            AstOp::Add => 0,
            AstOp::Sub => 1,
            AstOp::Mul => 2,
            AstOp::Div => 3,
            AstOp::TrueDiv => 4,
            AstOp::FloorDiv => 5,
            AstOp::Mod => 6,
            AstOp::PyMod => 7,
            AstOp::Pow => 8,
            AstOp::Equal => 9,
            AstOp::NullEqual => 10,
            AstOp::NotEqual => 11,
            AstOp::Less => 12,
            AstOp::Greater => 13,
            AstOp::LessEqual => 14,
            AstOp::GreaterEqual => 15,
            AstOp::BitwiseAnd => 16,
            AstOp::BitwiseOr => 17,
            AstOp::BitwiseXor => 18,
            AstOp::LogicalAnd => 19,
            AstOp::NullLogicalAnd => 20,
            AstOp::LogicalOr => 21,
            AstOp::NullLogicalOr => 22,
            AstOp::Identity => 23,
            AstOp::IsNull => 24,
            AstOp::Sin => 25,
            AstOp::Cos => 26,
            AstOp::Tan => 27,
            AstOp::Arcsin => 28,
            AstOp::Arccos => 29,
            AstOp::Arctan => 30,
            AstOp::Sinh => 31,
            AstOp::Cosh => 32,
            AstOp::Tanh => 33,
            AstOp::Arcsinh => 34,
            AstOp::Arccosh => 35,
            AstOp::Arctanh => 36,
            AstOp::Exp => 37,
            AstOp::Log => 38,
            AstOp::Sqrt => 39,
            AstOp::Cbrt => 40,
            AstOp::Ceil => 41,
            AstOp::Floor => 42,
            AstOp::Abs => 43,
            AstOp::Rint => 44,
            AstOp::BitInvert => 45,
            AstOp::Not => 46,
            AstOp::CastToInt64 => 47,
            AstOp::CastToUint64 => 48,
            AstOp::CastToFloat64 => 49,
        }
    }
}

/// An expression tree that can be evaluated per-row on the GPU.
///
/// `ExpressionTree` is the central type for building AST-based GPU
/// expressions. It owns a collection of expression nodes — literals,
/// column references, and operations — and provides methods to add each
/// kind.
///
/// After building, pass the tree and a root [`ExprRef`] to operations
/// like [`Table::compute_column`](crate::table::Table::compute_column)
/// or conditional join methods.
///
/// # Thread safety
///
/// An `ExpressionTree` is `Send` but not `Sync`. Build the tree on one
/// thread and pass it to GPU operations.
///
/// # Examples
///
/// ```ignore
/// use cudf::ast::{ExpressionTree, AstOp, TableSide};
///
/// let mut tree = ExpressionTree::new();
///
/// // Simple: col(0) > 10
/// let c = tree.col(0);
/// let lit = tree.lit_i32(10);
/// let pred = tree.gt(c, lit);
///
/// // For joins: left.col(0) == right.col(0)
/// let lc = tree.col_in(0, TableSide::Left);
/// let rc = tree.col_in(0, TableSide::Right);
/// let join_pred = tree.eq(lc, rc);
/// ```
#[doc(alias = "cudf::ast::tree")]
pub struct ExpressionTree(cxx::UniquePtr<cudf_sys::ast::ffi::ExpressionTree>);

impl ExpressionTree {
    /// Creates an empty expression tree with no nodes.
    ///
    /// Use methods like [`col`](Self::col), [`lit_i32`](Self::lit_i32), and
    /// [`add`](Self::add) to populate the tree.
    pub fn new() -> Self {
        Self(cudf_sys::ast::ffi::new_expression_tree())
    }

    /// Returns the number of expression nodes in the tree.
    pub fn len(&self) -> usize {
        cudf_sys::ast::ffi::expression_tree_len(&self.0)
    }

    /// Returns `true` if the tree contains no expression nodes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // -- Literals --

    /// Adds a 32-bit integer literal to the tree.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut tree = ExpressionTree::new();
    /// let lit = tree.lit_i32(42);
    /// ```
    pub fn lit_i32(&mut self, value: i32) -> ExprRef {
        let idx = cudf_sys::ast::ffi::expression_tree_add_literal_i32(self.0.pin_mut(), value);
        ExprRef(idx)
    }

    /// Adds a 64-bit integer literal to the tree.
    pub fn lit_i64(&mut self, value: i64) -> ExprRef {
        let idx = cudf_sys::ast::ffi::expression_tree_add_literal_i64(self.0.pin_mut(), value);
        ExprRef(idx)
    }

    /// Adds a 32-bit floating-point literal to the tree.
    pub fn lit_f32(&mut self, value: f32) -> ExprRef {
        let idx = cudf_sys::ast::ffi::expression_tree_add_literal_f32(self.0.pin_mut(), value);
        ExprRef(idx)
    }

    /// Adds a 64-bit floating-point literal to the tree.
    pub fn lit_f64(&mut self, value: f64) -> ExprRef {
        let idx = cudf_sys::ast::ffi::expression_tree_add_literal_f64(self.0.pin_mut(), value);
        ExprRef(idx)
    }

    /// Adds a boolean literal to the tree.
    pub fn lit_bool(&mut self, value: bool) -> ExprRef {
        let idx = cudf_sys::ast::ffi::expression_tree_add_literal_bool(self.0.pin_mut(), value);
        ExprRef(idx)
    }

    /// Adds a string literal to the tree.
    ///
    /// String literals can be used with string comparison operators.
    pub fn lit_str(&mut self, value: &str) -> ExprRef {
        let idx = cudf_sys::ast::ffi::expression_tree_add_literal_string(self.0.pin_mut(), value);
        ExprRef(idx)
    }

    // -- Column references --

    /// Adds a column reference by zero-based index, defaulting to the
    /// left table.
    ///
    /// For single-table operations like
    /// [`compute_column`](crate::table::Table::compute_column), all column
    /// references use the left (and only) table.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut tree = ExpressionTree::new();
    /// let first_col = tree.col(0);
    /// ```
    pub fn col(&mut self, index: usize) -> ExprRef {
        let idx = cudf_sys::ast::ffi::expression_tree_add_column_ref(
            self.0.pin_mut(),
            crate::usize_to_i32(index),
            TableSide::Left.to_i32(),
        );
        ExprRef(idx)
    }

    /// Adds a column reference with an explicit [`TableSide`].
    ///
    /// Use this for conditional join predicates where expressions refer to
    /// columns from both the left and right tables.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::ast::TableSide;
    ///
    /// let mut tree = ExpressionTree::new();
    /// let left_col = tree.col_in(0, TableSide::Left);
    /// let right_col = tree.col_in(0, TableSide::Right);
    /// let pred = tree.eq(left_col, right_col);
    /// ```
    pub fn col_in(&mut self, index: usize, side: TableSide) -> ExprRef {
        let idx = cudf_sys::ast::ffi::expression_tree_add_column_ref(
            self.0.pin_mut(),
            crate::usize_to_i32(index),
            side.to_i32(),
        );
        ExprRef(idx)
    }

    /// Adds a column reference by name.
    ///
    /// Named column references are useful for Parquet predicate pushdown
    /// where columns are identified by their schema name rather than index.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut tree = ExpressionTree::new();
    /// let col = tree.col_name("price");
    /// let threshold = tree.lit_f64(100.0);
    /// let pred = tree.gt(col, threshold);
    /// ```
    pub fn col_name(&mut self, name: &str) -> ExprRef {
        let idx = cudf_sys::ast::ffi::expression_tree_add_column_name_ref(self.0.pin_mut(), name);
        ExprRef(idx)
    }

    // -- Operations --

    /// Adds a unary operation node to the tree.
    ///
    /// # Arguments
    ///
    /// * `op` -- The [`AstOp`] to apply. Must be a unary operator (e.g.
    ///   [`Not`](AstOp::Not), [`Abs`](AstOp::Abs), [`IsNull`](AstOp::IsNull)).
    /// * `operand` -- The expression to apply the operator to.
    pub fn unary_op(&mut self, op: AstOp, operand: ExprRef) -> ExprRef {
        let idx = cudf_sys::ast::ffi::expression_tree_add_unary_op(
            self.0.pin_mut(),
            op.to_i32(),
            operand.0,
        );
        ExprRef(idx)
    }

    /// Adds a binary operation node to the tree.
    ///
    /// # Arguments
    ///
    /// * `op` -- The [`AstOp`] to apply. Must be a binary operator (e.g.
    ///   [`Add`](AstOp::Add), [`Equal`](AstOp::Equal)).
    /// * `left` -- The left operand expression.
    /// * `right` -- The right operand expression.
    pub fn binary_op(&mut self, op: AstOp, left: ExprRef, right: ExprRef) -> ExprRef {
        let idx = cudf_sys::ast::ffi::expression_tree_add_binary_op(
            self.0.pin_mut(),
            op.to_i32(),
            left.0,
            right.0,
        );
        ExprRef(idx)
    }

    // -- Convenience binary operators --

    /// Adds an addition (`left + right`) node.
    pub fn add(&mut self, left: ExprRef, right: ExprRef) -> ExprRef {
        self.binary_op(AstOp::Add, left, right)
    }

    /// Adds a subtraction (`left - right`) node.
    pub fn sub(&mut self, left: ExprRef, right: ExprRef) -> ExprRef {
        self.binary_op(AstOp::Sub, left, right)
    }

    /// Adds a multiplication (`left * right`) node.
    pub fn mul(&mut self, left: ExprRef, right: ExprRef) -> ExprRef {
        self.binary_op(AstOp::Mul, left, right)
    }

    /// Adds a division (`left / right`) node.
    pub fn div(&mut self, left: ExprRef, right: ExprRef) -> ExprRef {
        self.binary_op(AstOp::Div, left, right)
    }

    /// Adds an equality comparison (`left == right`) node.
    pub fn eq(&mut self, left: ExprRef, right: ExprRef) -> ExprRef {
        self.binary_op(AstOp::Equal, left, right)
    }

    /// Adds an inequality comparison (`left != right`) node.
    pub fn ne(&mut self, left: ExprRef, right: ExprRef) -> ExprRef {
        self.binary_op(AstOp::NotEqual, left, right)
    }

    /// Adds a less-than comparison (`left < right`) node.
    pub fn lt(&mut self, left: ExprRef, right: ExprRef) -> ExprRef {
        self.binary_op(AstOp::Less, left, right)
    }

    /// Adds a greater-than comparison (`left > right`) node.
    pub fn gt(&mut self, left: ExprRef, right: ExprRef) -> ExprRef {
        self.binary_op(AstOp::Greater, left, right)
    }

    /// Adds a less-than-or-equal comparison (`left <= right`) node.
    pub fn le(&mut self, left: ExprRef, right: ExprRef) -> ExprRef {
        self.binary_op(AstOp::LessEqual, left, right)
    }

    /// Adds a greater-than-or-equal comparison (`left >= right`) node.
    pub fn ge(&mut self, left: ExprRef, right: ExprRef) -> ExprRef {
        self.binary_op(AstOp::GreaterEqual, left, right)
    }

    /// Adds a logical AND (`left && right`) node.
    pub fn and(&mut self, left: ExprRef, right: ExprRef) -> ExprRef {
        self.binary_op(AstOp::LogicalAnd, left, right)
    }

    /// Adds a logical OR (`left || right`) node.
    pub fn or(&mut self, left: ExprRef, right: ExprRef) -> ExprRef {
        self.binary_op(AstOp::LogicalOr, left, right)
    }

    // -- Convenience unary operators --

    /// Adds a logical NOT (`!operand`) node.
    pub fn not(&mut self, operand: ExprRef) -> ExprRef {
        self.unary_op(AstOp::Not, operand)
    }

    /// Adds an IS NULL check node.
    pub fn is_null(&mut self, operand: ExprRef) -> ExprRef {
        self.unary_op(AstOp::IsNull, operand)
    }

    /// Adds an absolute value node.
    pub fn abs(&mut self, operand: ExprRef) -> ExprRef {
        self.unary_op(AstOp::Abs, operand)
    }

    /// Adds an arithmetic negation node (implemented as `0 - operand`).
    ///
    /// Uses the `Sub` operator with a zero literal to negate a value,
    /// since the C++ AST does not have a dedicated negation operator.
    pub fn neg(&mut self, operand: ExprRef) -> ExprRef {
        let zero = self.lit_i32(0);
        self.binary_op(AstOp::Sub, zero, operand)
    }

    /// Returns a reference to the underlying FFI expression tree.
    ///
    /// Used internally to pass the tree to cudf-sys functions.
    pub(crate) fn raw(&self) -> &cudf_sys::ast::ffi::ExpressionTree {
        &self.0
    }
}

impl Default for ExpressionTree {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::Column;
    use crate::stream::GpuOp;
    use crate::table::{Table, TableBuilder};

    /// Helper to build a two-column i32 table.
    fn make_two_col_table(col0: &[i32], col1: &[i32]) -> Table {
        let c0 = Column::from_slice_i32(col0).call().unwrap();
        let c1 = Column::from_slice_i32(col1).call().unwrap();
        let mut builder = TableBuilder::new();
        builder.push_column(c0);
        builder.push_column(c1);
        builder.build().unwrap()
    }

    #[test]
    fn tree_construction() {
        let mut tree = ExpressionTree::new();
        assert!(tree.is_empty());
        let a = tree.lit_i32(42);
        assert_eq!(tree.len(), 1);
        let b = tree.col(0);
        assert_eq!(tree.len(), 2);
        let c = tree.add(a, b);
        assert_eq!(tree.len(), 3);
        assert_eq!(c.index(), 2);
    }

    #[test]
    fn compute_column_add() {
        let c0 = Column::from_slice_i32(&[1, 2, 3]).call().unwrap();
        let c1 = Column::from_slice_i32(&[10, 20, 30]).call().unwrap();
        let mut builder = TableBuilder::new();
        builder.push_column(c0);
        builder.push_column(c1);
        let table = builder.build().unwrap();

        let mut tree = ExpressionTree::new();
        let a = tree.col(0);
        let b = tree.col(1);
        let sum = tree.add(a, b);

        let result = table.compute_column(&tree, sum).call().unwrap();
        assert_eq!(result.len(), 3);
        let vals: Vec<i32> = result.view().to_vec_i32().call().unwrap();
        assert_eq!(vals, vec![11, 22, 33]);
    }

    #[test]
    fn compute_column_comparison() {
        let c0 = Column::from_slice_i32(&[1, 5, 10, 15]).call().unwrap();
        let mut builder = TableBuilder::new();
        builder.push_column(c0);
        let table = builder.build().unwrap();

        let mut tree = ExpressionTree::new();
        let col = tree.col(0);
        let threshold = tree.lit_i32(8);
        let pred = tree.gt(col, threshold);

        let result = table.compute_column(&tree, pred).call().unwrap();
        assert_eq!(result.len(), 4);
        // Should be [false, false, true, true]
        let vals: Vec<bool> = result.view().to_vec_bool().call().unwrap();
        assert_eq!(vals, vec![false, false, true, true]);
    }

    #[test]
    fn conditional_inner_join_basic() {
        let left = make_two_col_table(&[1, 2, 3], &[10, 20, 30]);
        let right = make_two_col_table(&[2, 3, 4], &[200, 300, 400]);

        // Join on left.col(0) == right.col(0)
        let mut tree = ExpressionTree::new();
        let lc = tree.col_in(0, TableSide::Left);
        let rc = tree.col_in(0, TableSide::Right);
        let pred = tree.eq(lc, rc);

        let result = left
            .conditional_inner_join(&right, &tree, pred)
            .call()
            .unwrap();
        assert_eq!(result.columns_len(), 4);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn nested_expression() {
        let table = make_two_col_table(&[1, 2, 3], &[10, 20, 30]);

        // (col(0) + col(1)) > lit(25)
        let mut tree = ExpressionTree::new();
        let a = tree.col(0);
        let b = tree.col(1);
        let sum = tree.add(a, b);
        let threshold = tree.lit_i32(25);
        let pred = tree.gt(sum, threshold);

        let result = table.compute_column(&tree, pred).call().unwrap();
        let vals: Vec<bool> = result.view().to_vec_bool().call().unwrap();
        assert_eq!(vals, vec![false, false, true]);
    }

    #[test]
    fn tree_default() {
        let tree = ExpressionTree::default();
        assert!(tree.is_empty());
    }

    #[test]
    fn expr_ref_debug_copy() {
        let mut tree = ExpressionTree::new();
        let r = tree.lit_i32(1);
        let r2 = r; // Copy
        assert_eq!(r, r2);
        assert_eq!(format!("{r:?}"), "ExprRef(0)");
    }

    #[test]
    fn ast_op_to_i32_boundaries() {
        assert_eq!(AstOp::Add.to_i32(), 0);
        assert_eq!(AstOp::CastToFloat64.to_i32(), 49);
    }
}
