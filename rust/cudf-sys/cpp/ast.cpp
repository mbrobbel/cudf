// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/ast.rs.h"
#include "cudf-sys/src/lib.rs.h"
#include "cudf-sys/helpers.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/ast/ast_operator.hpp>
#include <cudf/transform.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/join/conditional_join.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/copying.hpp>

namespace cudf_sys {

// Helper to convert index vectors to columns (same pattern as join.cpp).
static std::unique_ptr<cudf::column> idx_to_column(
    std::unique_ptr<rmm::device_uvector<cudf::size_type>> indices) {
  auto size = static_cast<cudf::size_type>(indices->size());
  auto buf = indices->release();
  return std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT32}, size, std::move(buf),
      rmm::device_buffer{}, 0);
}

// -- ExpressionTree builder functions --

std::unique_ptr<ExpressionTree> new_expression_tree() {
  return std::make_unique<ExpressionTree>();
}

std::size_t expression_tree_len(ExpressionTree const& tree) {
  return tree.len();
}

std::size_t expression_tree_add_literal_i32(ExpressionTree& tree, int32_t value) {
  auto s = std::make_unique<cudf::numeric_scalar<int32_t>>(value, true);
  auto& ref = *s;
  tree.scalars().push_back(std::move(s));
  tree.tree().emplace<cudf::ast::literal>(ref);
  return tree.len() - 1;
}

std::size_t expression_tree_add_literal_i64(ExpressionTree& tree, int64_t value) {
  auto s = std::make_unique<cudf::numeric_scalar<int64_t>>(value, true);
  auto& ref = *s;
  tree.scalars().push_back(std::move(s));
  tree.tree().emplace<cudf::ast::literal>(ref);
  return tree.len() - 1;
}

std::size_t expression_tree_add_literal_f32(ExpressionTree& tree, float value) {
  auto s = std::make_unique<cudf::numeric_scalar<float>>(value, true);
  auto& ref = *s;
  tree.scalars().push_back(std::move(s));
  tree.tree().emplace<cudf::ast::literal>(ref);
  return tree.len() - 1;
}

std::size_t expression_tree_add_literal_f64(ExpressionTree& tree, double value) {
  auto s = std::make_unique<cudf::numeric_scalar<double>>(value, true);
  auto& ref = *s;
  tree.scalars().push_back(std::move(s));
  tree.tree().emplace<cudf::ast::literal>(ref);
  return tree.len() - 1;
}

std::size_t expression_tree_add_literal_bool(ExpressionTree& tree, bool value) {
  auto s = std::make_unique<cudf::numeric_scalar<bool>>(value, true);
  auto& ref = *s;
  tree.scalars().push_back(std::move(s));
  tree.tree().emplace<cudf::ast::literal>(ref);
  return tree.len() - 1;
}

std::size_t expression_tree_add_literal_string(ExpressionTree& tree, rust::Str value) {
  auto s = std::make_unique<cudf::string_scalar>(std::string(value.data(), value.size()), true);
  auto& ref = *s;
  tree.scalars().push_back(std::move(s));
  tree.tree().emplace<cudf::ast::literal>(ref);
  return tree.len() - 1;
}

std::size_t expression_tree_add_column_ref(ExpressionTree& tree, int32_t column_index, int32_t table_source) {
  tree.tree().emplace<cudf::ast::column_reference>(
      column_index, static_cast<cudf::ast::table_reference>(table_source));
  return tree.len() - 1;
}

std::size_t expression_tree_add_column_name_ref(ExpressionTree& tree, rust::Str name) {
  tree.tree().emplace<cudf::ast::column_name_reference>(
      std::string(name.data(), name.size()));
  return tree.len() - 1;
}

std::size_t expression_tree_add_unary_op(ExpressionTree& tree, int32_t op, std::size_t operand) {
  tree.tree().emplace<cudf::ast::operation>(
      static_cast<cudf::ast::ast_operator>(op), tree.at(operand));
  return tree.len() - 1;
}

std::size_t expression_tree_add_binary_op(ExpressionTree& tree, int32_t op, std::size_t left, std::size_t right) {
  tree.tree().emplace<cudf::ast::operation>(
      static_cast<cudf::ast::ast_operator>(op), tree.at(left), tree.at(right));
  return tree.len() - 1;
}

// -- compute_column --

std::unique_ptr<Column> ast_compute_column(
    Table const& tbl, ExpressionTree const& tree, std::size_t root_index, std::size_t stream) {
  return COL(cudf::compute_column(tbl.cached_view(), tree.at(root_index), S(stream)));
}

// -- Parquet filter --

std::unique_ptr<Table> read_parquet_filtered(
    rust::Str filepath, ExpressionTree const& tree, std::size_t root_index) {
  std::string path(filepath.data(), filepath.size());
  auto opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{path})
      .filter(tree.at(root_index))
      .build();
  return TBL(cudf::io::read_parquet(opts).tbl);
}

// -- Conditional joins --

std::unique_ptr<Table> conditional_inner_join(
    Table const& left, Table const& right,
    ExpressionTree const& tree, std::size_t root_index,
    std::size_t stream) {
  auto s = S(stream);
  auto lv = left.cached_view();
  auto rv = right.cached_view();
  auto [li, ri] = cudf::conditional_inner_join(lv, rv, tree.at(root_index), {}, s);
  auto lic = idx_to_column(std::move(li));
  auto ric = idx_to_column(std::move(ri));
  auto lr = cudf::gather(lv, lic->view(), cudf::out_of_bounds_policy::DONT_CHECK, s);
  auto rr = cudf::gather(rv, ric->view(), cudf::out_of_bounds_policy::NULLIFY, s);
  auto lc = lr->release();
  auto rc = rr->release();
  std::vector<std::unique_ptr<cudf::column>> all;
  all.reserve(lc.size() + rc.size());
  for (auto& c : lc) all.push_back(std::move(c));
  for (auto& c : rc) all.push_back(std::move(c));
  return TBL(std::make_unique<cudf::table>(std::move(all)));
}

std::unique_ptr<Table> conditional_left_join(
    Table const& left, Table const& right,
    ExpressionTree const& tree, std::size_t root_index,
    std::size_t stream) {
  auto s = S(stream);
  auto lv = left.cached_view();
  auto rv = right.cached_view();
  auto [li, ri] = cudf::conditional_left_join(lv, rv, tree.at(root_index), {}, s);
  auto lic = idx_to_column(std::move(li));
  auto ric = idx_to_column(std::move(ri));
  auto lr = cudf::gather(lv, lic->view(), cudf::out_of_bounds_policy::DONT_CHECK, s);
  auto rr = cudf::gather(rv, ric->view(), cudf::out_of_bounds_policy::NULLIFY, s);
  auto lc = lr->release();
  auto rc = rr->release();
  std::vector<std::unique_ptr<cudf::column>> all;
  all.reserve(lc.size() + rc.size());
  for (auto& c : lc) all.push_back(std::move(c));
  for (auto& c : rc) all.push_back(std::move(c));
  return TBL(std::make_unique<cudf::table>(std::move(all)));
}

std::unique_ptr<Table> conditional_full_join(
    Table const& left, Table const& right,
    ExpressionTree const& tree, std::size_t root_index,
    std::size_t stream) {
  auto s = S(stream);
  auto lv = left.cached_view();
  auto rv = right.cached_view();
  auto [li, ri] = cudf::conditional_full_join(lv, rv, tree.at(root_index), s);
  auto lic = idx_to_column(std::move(li));
  auto ric = idx_to_column(std::move(ri));
  auto lr = cudf::gather(lv, lic->view(), cudf::out_of_bounds_policy::NULLIFY, s);
  auto rr = cudf::gather(rv, ric->view(), cudf::out_of_bounds_policy::NULLIFY, s);
  auto lc = lr->release();
  auto rc = rr->release();
  std::vector<std::unique_ptr<cudf::column>> all;
  all.reserve(lc.size() + rc.size());
  for (auto& c : lc) all.push_back(std::move(c));
  for (auto& c : rc) all.push_back(std::move(c));
  return TBL(std::make_unique<cudf::table>(std::move(all)));
}

std::unique_ptr<Table> conditional_left_semi_join(
    Table const& left, Table const& right,
    ExpressionTree const& tree, std::size_t root_index,
    std::size_t stream) {
  auto s = S(stream);
  auto lv = left.cached_view();
  auto idx = cudf::conditional_left_semi_join(lv, right.cached_view(), tree.at(root_index), {}, s);
  auto ic = idx_to_column(std::move(idx));
  return TBL(cudf::gather(lv, ic->view(), cudf::out_of_bounds_policy::DONT_CHECK, s));
}

std::unique_ptr<Table> conditional_left_anti_join(
    Table const& left, Table const& right,
    ExpressionTree const& tree, std::size_t root_index,
    std::size_t stream) {
  auto s = S(stream);
  auto lv = left.cached_view();
  auto idx = cudf::conditional_left_anti_join(lv, right.cached_view(), tree.at(root_index), {}, s);
  auto ic = idx_to_column(std::move(idx));
  return TBL(cudf::gather(lv, ic->view(), cudf::out_of_bounds_policy::DONT_CHECK, s));
}

}  // namespace cudf_sys
