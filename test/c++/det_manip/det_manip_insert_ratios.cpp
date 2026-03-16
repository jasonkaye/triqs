// Copyright (c) 2024 Simons Foundation
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You may obtain a copy of the License at
//     https://www.gnu.org/licenses/gpl-3.0.txt

#include <triqs/det_manip/det_manip.hpp>
#include <triqs/det_manip/det_manip_basic.hpp>
#include <triqs/mc_tools/random_generator.hpp>
#include <nda/linalg/det.hpp>
#include <iostream>
#include <cmath>

struct fun {
  using result_type   = double;
  using argument_type = double;

  double operator()(double x, double y) const {
    const double pi   = acos(-1.0);
    const double beta = 10.0;
    const double epsi = 0.1;
    double tau        = x - y;
    bool s            = (tau > 0);
    tau               = (s ? tau : beta + tau);
    double r          = epsi + tau / beta * (1 - 2 * epsi);
    return -2 * (pi / beta) / std::sin(pi * r);
  }
};

// Relative tolerance for comparing gemm-based batch vs gemv-based sequential results.
// gemm and gemv use different evaluation orders in BLAS, leading to O(N*kappa*eps) differences
// that can be amplified by near-cancellation in the Schur complement.
const double PRECISION = 1.e-6;

template <typename T1, typename T2> void assert_close(T1 const &A, T2 const &B, double precision, std::string const &msg = "") {
  double diff  = std::abs(A - B);
  double scale = std::max(std::abs(double(A)), std::abs(double(B)));
  if (diff > precision * std::max(scale, 1.0))
    TRIQS_RUNTIME_ERROR << "assert_close error: " << A << " vs " << B << " diff=" << diff << " reldiff=" << diff / std::max(scale, 1e-30) << " " << msg;
}

// Build up a det_manip of given size
template <typename DM> void build_det(DM &D, int target_size, triqs::mc_tools::random_generator &RNG) {
  for (int n = 0; n < target_size; ++n) {
    double x = RNG(10.0);
    double y = RNG(10.0);
    D.insert(D.size(), D.size(), x, y);
  }
}

// Helper: fill an nda::array<double, 1> with random values
nda::array<double, 1> random_array1(long K, triqs::mc_tools::random_generator &RNG, double range = 10.0) {
  nda::array<double, 1> a(K);
  for (long m = 0; m < K; ++m) a(m) = RNG(range);
  return a;
}

// Helper: fill an nda::array<double, 2> with random values
nda::array<double, 2> random_array2(long M, long N, triqs::mc_tools::random_generator &RNG, double range = 10.0) {
  nda::array<double, 2> a(M, N);
  for (long i = 0; i < M; ++i)
    for (long j = 0; j < N; ++j) a(i, j) = RNG(range);
  return a;
}

void test_rank1_batch_vs_sequential() {
  std::cerr << "=== test_rank1_batch_vs_sequential ===" << std::endl;
  fun f;
  triqs::det_manip::det_manip<fun> D(f, 100);
  triqs::mc_tools::random_generator RNG("mt19937", 12345);
  build_det(D, 20, RNG);

  long K = 30;
  auto xs = random_array1(K, RNG);
  auto ys = random_array1(K, RNG);

  auto batch = D.insert_ratios(0, 0, xs, ys);

  for (long m = 0; m < K; ++m) {
    auto ratio = D.try_insert(0, 0, xs(m), ys(m));
    D.reject_last_try();
    assert_close(batch(m), ratio, PRECISION, "rank1 m=" + std::to_string(m));
  }
  std::cerr << "PASSED" << std::endl;
}

void test_rank1_state_unchanged() {
  std::cerr << "=== test_rank1_state_unchanged ===" << std::endl;
  fun f;
  triqs::det_manip::det_manip<fun> D(f, 100);
  triqs::mc_tools::random_generator RNG("mt19937", 54321);
  build_det(D, 15, RNG);

  auto det_before  = D.determinant();
  auto inv_before  = D.inverse_matrix();
  auto size_before = D.size();

  auto xs = nda::array<double, 1>{1.0, 2.0, 3.0};
  auto ys = nda::array<double, 1>{4.0, 5.0, 6.0};
  D.insert_ratios(0, 0, xs, ys);

  if (D.size() != size_before) TRIQS_RUNTIME_ERROR << "Size changed!";
  assert_close(D.determinant(), det_before, 1.e-12, "det changed");
  auto inv_after = D.inverse_matrix();
  for (int i = 0; i < size_before; ++i)
    for (int j = 0; j < size_before; ++j) assert_close(inv_after(i, j), inv_before(i, j), 1.e-12, "inv changed");
  std::cerr << "PASSED" << std::endl;
}

void test_rank1_empty_matrix() {
  std::cerr << "=== test_rank1_empty_matrix ===" << std::endl;
  fun f;
  triqs::det_manip::det_manip<fun> D(f, 100);

  auto xs = nda::array<double, 1>{1.0, 2.0, 3.0};
  auto ys = nda::array<double, 1>{4.0, 5.0, 6.0};
  auto batch = D.insert_ratios(0, 0, xs, ys);

  for (long m = 0; m < 3; ++m) {
    auto ratio = D.try_insert(0, 0, xs(m), ys(m));
    D.reject_last_try();
    assert_close(batch(m), ratio, 1.e-14, "rank1 empty m=" + std::to_string(m));
  }
  std::cerr << "PASSED" << std::endl;
}

void test_rank1_single_point() {
  std::cerr << "=== test_rank1_single_point ===" << std::endl;
  fun f;
  triqs::det_manip::det_manip<fun> D(f, 100);
  triqs::mc_tools::random_generator RNG("mt19937", 99999);
  build_det(D, 10, RNG);

  auto xs = nda::array<double, 1>{5.0};
  auto ys = nda::array<double, 1>{3.0};
  auto batch = D.insert_ratios(0, 0, xs, ys);
  auto ratio = D.try_insert(0, 0, xs(0), ys(0));
  D.reject_last_try();
  assert_close(batch(0), ratio, PRECISION, "rank1 single");
  std::cerr << "PASSED" << std::endl;
}

void test_rank1_empty_batch() {
  std::cerr << "=== test_rank1_empty_batch ===" << std::endl;
  fun f;
  triqs::det_manip::det_manip<fun> D(f, 100);
  triqs::mc_tools::random_generator RNG("mt19937", 11111);
  build_det(D, 5, RNG);

  nda::array<double, 1> xs(0), ys(0);
  auto batch = D.insert_ratios(0, 0, xs, ys);
  if (batch.size() != 0) TRIQS_RUNTIME_ERROR << "Expected empty result";
  std::cerr << "PASSED" << std::endl;
}

void test_rank2_batch_vs_sequential() {
  std::cerr << "=== test_rank2_batch_vs_sequential ===" << std::endl;
  fun f;
  triqs::det_manip::det_manip<fun> D(f, 100);
  triqs::mc_tools::random_generator RNG("mt19937", 67890);
  build_det(D, 20, RNG);

  long K = 30;
  auto x0s = random_array1(K, RNG);
  auto x1s = random_array1(K, RNG);
  auto y0s = random_array1(K, RNG);
  auto y1s = random_array1(K, RNG);

  auto batch = D.insert2_ratios(0, 1, 0, 1, x0s, x1s, y0s, y1s);

  for (long m = 0; m < K; ++m) {
    auto ratio = D.try_insert2(0, 1, 0, 1, x0s(m), x1s(m), y0s(m), y1s(m));
    D.reject_last_try();
    assert_close(batch(m), ratio, PRECISION, "rank2 m=" + std::to_string(m));
  }
  std::cerr << "PASSED" << std::endl;
}

void test_rank2_state_unchanged() {
  std::cerr << "=== test_rank2_state_unchanged ===" << std::endl;
  fun f;
  triqs::det_manip::det_manip<fun> D(f, 100);
  triqs::mc_tools::random_generator RNG("mt19937", 22222);
  build_det(D, 15, RNG);

  auto det_before  = D.determinant();
  auto size_before = D.size();

  auto x0s = nda::array<double, 1>{1.0, 2.0};
  auto x1s = nda::array<double, 1>{3.0, 4.0};
  auto y0s = nda::array<double, 1>{5.0, 6.0};
  auto y1s = nda::array<double, 1>{7.0, 8.0};
  D.insert2_ratios(0, 1, 0, 1, x0s, x1s, y0s, y1s);

  if (D.size() != size_before) TRIQS_RUNTIME_ERROR << "Size changed!";
  assert_close(D.determinant(), det_before, 1.e-12, "det changed");
  std::cerr << "PASSED" << std::endl;
}

void test_rank2_empty_matrix() {
  std::cerr << "=== test_rank2_empty_matrix ===" << std::endl;
  fun f;
  triqs::det_manip::det_manip<fun> D(f, 100);

  auto x0s = nda::array<double, 1>{1.0, 2.0};
  auto x1s = nda::array<double, 1>{3.0, 4.0};
  auto y0s = nda::array<double, 1>{5.0, 6.0};
  auto y1s = nda::array<double, 1>{7.0, 8.0};
  auto batch = D.insert2_ratios(0, 1, 0, 1, x0s, x1s, y0s, y1s);

  for (long m = 0; m < 2; ++m) {
    auto ratio = D.try_insert2(0, 1, 0, 1, x0s(m), x1s(m), y0s(m), y1s(m));
    D.reject_last_try();
    assert_close(batch(m), ratio, 1.e-14, "rank2 empty m=" + std::to_string(m));
  }
  std::cerr << "PASSED" << std::endl;
}

void test_rank2_reversed_indices() {
  std::cerr << "=== test_rank2_reversed_indices ===" << std::endl;
  fun f;
  triqs::det_manip::det_manip<fun> D(f, 100);
  triqs::mc_tools::random_generator RNG("mt19937", 55555);
  build_det(D, 20, RNG);

  long K = 15;
  auto x0s = random_array1(K, RNG);
  auto x1s = random_array1(K, RNG);
  auto y0s = random_array1(K, RNG);
  auto y1s = random_array1(K, RNG);

  // Test i0 > i1 (reversed row indices)
  auto batch_10 = D.insert2_ratios(1, 0, 0, 1, x0s, x1s, y0s, y1s);
  for (long m = 0; m < K; ++m) {
    auto ratio = D.try_insert2(1, 0, 0, 1, x0s(m), x1s(m), y0s(m), y1s(m));
    D.reject_last_try();
    assert_close(batch_10(m), ratio, PRECISION, "rank2 reversed i m=" + std::to_string(m));
  }

  // Test j0 > j1 (reversed col indices)
  auto batch_01 = D.insert2_ratios(0, 1, 1, 0, x0s, x1s, y0s, y1s);
  for (long m = 0; m < K; ++m) {
    auto ratio = D.try_insert2(0, 1, 1, 0, x0s(m), x1s(m), y0s(m), y1s(m));
    D.reject_last_try();
    assert_close(batch_01(m), ratio, PRECISION, "rank2 reversed j m=" + std::to_string(m));
  }

  // Test both reversed
  auto batch_11 = D.insert2_ratios(1, 0, 1, 0, x0s, x1s, y0s, y1s);
  for (long m = 0; m < K; ++m) {
    auto ratio = D.try_insert2(1, 0, 1, 0, x0s(m), x1s(m), y0s(m), y1s(m));
    D.reject_last_try();
    assert_close(batch_11(m), ratio, PRECISION, "rank2 both reversed m=" + std::to_string(m));
  }

  std::cerr << "PASSED" << std::endl;
}

void test_cross_validate_rank1() {
  std::cerr << "=== test_cross_validate_rank1 ===" << std::endl;
  fun f;
  triqs::det_manip::det_manip<fun> D(f, 100);
  triqs::det_manip::det_manip_basic<fun> Db(f, 100);
  triqs::mc_tools::random_generator RNG("mt19937", 33333);

  for (int n = 0; n < 15; ++n) {
    double x = RNG(10.0);
    double y = RNG(10.0);
    D.insert(D.size(), D.size(), x, y);
    Db.insert(Db.size(), Db.size(), x, y);
  }

  long K = 10;
  auto xs = random_array1(K, RNG);
  auto ys = random_array1(K, RNG);

  auto batch_opt   = D.insert_ratios(0, 0, xs, ys);
  auto batch_basic = Db.insert_ratios(0, 0, xs, ys);

  for (long m = 0; m < K; ++m) assert_close(batch_opt(m), batch_basic(m), 1.e-4, "cross rank1 m=" + std::to_string(m));
  std::cerr << "PASSED" << std::endl;
}

void test_cross_validate_rank2() {
  std::cerr << "=== test_cross_validate_rank2 ===" << std::endl;
  fun f;
  triqs::det_manip::det_manip<fun> D(f, 100);
  triqs::det_manip::det_manip_basic<fun> Db(f, 100);
  triqs::mc_tools::random_generator RNG("mt19937", 44444);

  for (int n = 0; n < 15; ++n) {
    double x = RNG(10.0);
    double y = RNG(10.0);
    D.insert(D.size(), D.size(), x, y);
    Db.insert(Db.size(), Db.size(), x, y);
  }

  long K = 10;
  auto x0s = random_array1(K, RNG);
  auto x1s = random_array1(K, RNG);
  auto y0s = random_array1(K, RNG);
  auto y1s = random_array1(K, RNG);

  auto batch_opt   = D.insert2_ratios(0, 1, 0, 1, x0s, x1s, y0s, y1s);
  auto batch_basic = Db.insert2_ratios(0, 1, 0, 1, x0s, x1s, y0s, y1s);

  for (long m = 0; m < K; ++m) assert_close(batch_opt(m), batch_basic(m), 1.e-3, "cross rank2 m=" + std::to_string(m));
  std::cerr << "PASSED" << std::endl;
}

void test_matrix_vs_sequential() {
  std::cerr << "=== test_matrix_vs_sequential ===" << std::endl;
  fun f;
  triqs::det_manip::det_manip<fun> D(f, 100);
  triqs::mc_tools::random_generator RNG("mt19937", 77777);
  build_det(D, 20, RNG);

  long Kx = 8, Ky = 10;
  auto xs = random_array1(Kx, RNG);
  auto ys = random_array1(Ky, RNG);

  auto mat = D.insert_ratios_matrix(0, 0, xs, ys);

  for (long a = 0; a < Kx; ++a)
    for (long b = 0; b < Ky; ++b) {
      auto ratio = D.try_insert(0, 0, xs(a), ys(b));
      D.reject_last_try();
      assert_close(mat(a, b), ratio, 1.e-5, "matrix (" + std::to_string(a) + "," + std::to_string(b) + ")");
    }
  std::cerr << "PASSED" << std::endl;
}

void test_matrix_state_unchanged() {
  std::cerr << "=== test_matrix_state_unchanged ===" << std::endl;
  fun f;
  triqs::det_manip::det_manip<fun> D(f, 100);
  triqs::mc_tools::random_generator RNG("mt19937", 88888);
  build_det(D, 15, RNG);

  auto det_before  = D.determinant();
  auto inv_before  = D.inverse_matrix();
  auto size_before = D.size();

  auto xs = nda::array<double, 1>{1.0, 2.0, 3.0};
  auto ys = nda::array<double, 1>{4.0, 5.0};
  D.insert_ratios_matrix(0, 0, xs, ys);

  if (D.size() != size_before) TRIQS_RUNTIME_ERROR << "Size changed!";
  assert_close(D.determinant(), det_before, 1.e-12, "det changed");
  auto inv_after = D.inverse_matrix();
  for (int i = 0; i < size_before; ++i)
    for (int j = 0; j < size_before; ++j) assert_close(inv_after(i, j), inv_before(i, j), 1.e-12, "inv changed");
  std::cerr << "PASSED" << std::endl;
}

void test_matrix_empty() {
  std::cerr << "=== test_matrix_empty ===" << std::endl;
  fun f;
  triqs::det_manip::det_manip<fun> D(f, 100);

  auto xs = nda::array<double, 1>{1.0, 2.0};
  auto ys = nda::array<double, 1>{3.0, 4.0, 5.0};
  auto mat = D.insert_ratios_matrix(0, 0, xs, ys);

  for (long a = 0; a < 2; ++a)
    for (long b = 0; b < 3; ++b) {
      auto ratio = D.try_insert(0, 0, xs(a), ys(b));
      D.reject_last_try();
      assert_close(mat(a, b), ratio, 1.e-14, "matrix empty (" + std::to_string(a) + "," + std::to_string(b) + ")");
    }
  std::cerr << "PASSED" << std::endl;
}

void test_matrix_cross_validate() {
  std::cerr << "=== test_matrix_cross_validate ===" << std::endl;
  fun f;
  triqs::det_manip::det_manip<fun> D(f, 100);
  triqs::det_manip::det_manip_basic<fun> Db(f, 100);
  triqs::mc_tools::random_generator RNG("mt19937", 66666);

  for (int n = 0; n < 15; ++n) {
    double x = RNG(10.0);
    double y = RNG(10.0);
    D.insert(D.size(), D.size(), x, y);
    Db.insert(Db.size(), Db.size(), x, y);
  }

  long Kx = 6, Ky = 8;
  auto xs = random_array1(Kx, RNG);
  auto ys = random_array1(Ky, RNG);

  auto mat_opt   = D.insert_ratios_matrix(0, 0, xs, ys);
  auto mat_basic = Db.insert_ratios_matrix(0, 0, xs, ys);

  for (long a = 0; a < Kx; ++a)
    for (long b = 0; b < Ky; ++b)
      assert_close(mat_opt(a, b), mat_basic(a, b), 1.e-4, "cross matrix (" + std::to_string(a) + "," + std::to_string(b) + ")");
  std::cerr << "PASSED" << std::endl;
}

void test_matrix_vs_paired() {
  std::cerr << "=== test_matrix_vs_paired ===" << std::endl;
  fun f;
  triqs::det_manip::det_manip<fun> D(f, 100);
  triqs::mc_tools::random_generator RNG("mt19937", 99998);
  build_det(D, 20, RNG);

  long K = 15;
  auto xs = random_array1(K, RNG);
  auto ys = random_array1(K, RNG);

  auto mat    = D.insert_ratios_matrix(0, 0, xs, ys);
  auto paired = D.insert_ratios(0, 0, xs, ys);

  for (long m = 0; m < K; ++m)
    assert_close(mat(m, m), paired(m), PRECISION, "matrix diagonal m=" + std::to_string(m));
  std::cerr << "PASSED" << std::endl;
}

// ---- Tests for rank-2 arrays (batching across two dimensions) ----

void test_rank2_array_insert_ratios() {
  std::cerr << "=== test_rank2_array_insert_ratios ===" << std::endl;
  fun f;
  triqs::det_manip::det_manip<fun> D(f, 100);
  triqs::mc_tools::random_generator RNG("mt19937", 11223);
  build_det(D, 20, RNG);

  long M = 5, E = 8;
  auto xs = random_array2(M, E, RNG);
  auto ys = random_array2(M, E, RNG);

  auto batch = D.insert_ratios(0, 0, xs, ys);
  static_assert(nda::get_rank<decltype(batch)> == 2);
  TRIQS_ASSERT(batch.shape() == (std::array<long, 2>{M, E}));

  // Cross-validate against sequential calls
  for (long i = 0; i < M; ++i)
    for (long j = 0; j < E; ++j) {
      auto ratio = D.try_insert(0, 0, xs(i, j), ys(i, j));
      D.reject_last_try();
      assert_close(batch(i, j), ratio, PRECISION, "rank2 array (" + std::to_string(i) + "," + std::to_string(j) + ")");
    }
  std::cerr << "PASSED" << std::endl;
}

void test_rank2_array_insert_ratios_cross_validate() {
  std::cerr << "=== test_rank2_array_insert_ratios_cross_validate ===" << std::endl;
  fun f;
  triqs::det_manip::det_manip<fun> D(f, 100);
  triqs::det_manip::det_manip_basic<fun> Db(f, 100);
  triqs::mc_tools::random_generator RNG("mt19937", 44556);

  for (int n = 0; n < 15; ++n) {
    double x = RNG(10.0);
    double y = RNG(10.0);
    D.insert(D.size(), D.size(), x, y);
    Db.insert(Db.size(), Db.size(), x, y);
  }

  long M = 4, E = 6;
  auto xs = random_array2(M, E, RNG);
  auto ys = random_array2(M, E, RNG);

  auto batch_opt   = D.insert_ratios(0, 0, xs, ys);
  auto batch_basic = Db.insert_ratios(0, 0, xs, ys);

  for (long i = 0; i < M; ++i)
    for (long j = 0; j < E; ++j)
      assert_close(batch_opt(i, j), batch_basic(i, j), 1.e-2,
                   "cross rank2 array (" + std::to_string(i) + "," + std::to_string(j) + ")");
  std::cerr << "PASSED" << std::endl;
}

void test_rank2_array_insert2_ratios() {
  std::cerr << "=== test_rank2_array_insert2_ratios ===" << std::endl;
  fun f;
  triqs::det_manip::det_manip<fun> D(f, 100);
  triqs::mc_tools::random_generator RNG("mt19937", 55667);
  build_det(D, 20, RNG);

  long M = 5, E = 8;
  auto x0s = random_array2(M, E, RNG);
  auto x1s = random_array2(M, E, RNG);
  auto y0s = random_array2(M, E, RNG);
  auto y1s = random_array2(M, E, RNG);

  auto batch = D.insert2_ratios(0, 1, 0, 1, x0s, x1s, y0s, y1s);
  static_assert(nda::get_rank<decltype(batch)> == 2);
  TRIQS_ASSERT(batch.shape() == (std::array<long, 2>{M, E}));

  for (long i = 0; i < M; ++i)
    for (long j = 0; j < E; ++j) {
      auto ratio = D.try_insert2(0, 1, 0, 1, x0s(i, j), x1s(i, j), y0s(i, j), y1s(i, j));
      D.reject_last_try();
      assert_close(batch(i, j), ratio, PRECISION,
                   "rank2 array insert2 (" + std::to_string(i) + "," + std::to_string(j) + ")");
    }
  std::cerr << "PASSED" << std::endl;
}

// ---- Tests for broadcast: insert2_ratios with mixed rank-2 / rank-1 args ----

void test_broadcast_insert2_ratios() {
  std::cerr << "=== test_broadcast_insert2_ratios ===" << std::endl;
  fun f;
  triqs::det_manip::det_manip<fun> D(f, 100);
  triqs::mc_tools::random_generator RNG("mt19937", 77889);
  build_det(D, 20, RNG);

  long M = 5, E = 8;
  // Pair 0 (A-side): rank-2, shape (M, E)
  auto x0s = random_array2(M, E, RNG);
  auto y0s = random_array2(M, E, RNG);
  // Pair 1 (B-side): rank-1, shape (E) -- broadcast across M
  auto x1s = random_array1(E, RNG);
  auto y1s = random_array1(E, RNG);

  auto batch = D.insert2_ratios(0, 1, 0, 1, x0s, x1s, y0s, y1s);
  static_assert(nda::get_rank<decltype(batch)> == 2);
  TRIQS_ASSERT(batch.shape() == (std::array<long, 2>{M, E}));

  // Cross-validate against sequential calls
  for (long i = 0; i < M; ++i)
    for (long j = 0; j < E; ++j) {
      auto ratio = D.try_insert2(0, 1, 0, 1, x0s(i, j), x1s(j), y0s(i, j), y1s(j));
      D.reject_last_try();
      assert_close(batch(i, j), ratio, PRECISION,
                   "broadcast insert2 (" + std::to_string(i) + "," + std::to_string(j) + ")");
    }
  std::cerr << "PASSED" << std::endl;
}

void test_broadcast_insert2_ratios_basic_vs_sequential() {
  std::cerr << "=== test_broadcast_insert2_ratios_basic_vs_sequential ===" << std::endl;
  fun f;
  triqs::det_manip::det_manip_basic<fun> Db(f, 100);
  triqs::mc_tools::random_generator RNG("mt19937", 88990);

  for (int n = 0; n < 15; ++n) {
    double x = RNG(10.0);
    double y = RNG(10.0);
    Db.insert(Db.size(), Db.size(), x, y);
  }

  long M = 4, E = 6;
  auto x0s = random_array2(M, E, RNG);
  auto y0s = random_array2(M, E, RNG);
  auto x1s = random_array1(E, RNG);
  auto y1s = random_array1(E, RNG);

  auto batch_basic = Db.insert2_ratios(0, 1, 0, 1, x0s, x1s, y0s, y1s);

  for (long i = 0; i < M; ++i)
    for (long j = 0; j < E; ++j) {
      auto ratio = Db.try_insert2(0, 1, 0, 1, x0s(i, j), x1s(j), y0s(i, j), y1s(j));
      Db.reject_last_try();
      assert_close(batch_basic(i, j), ratio, 1.e-10,
                   "basic broadcast vs sequential (" + std::to_string(i) + "," + std::to_string(j) + ")");
    }
  std::cerr << "PASSED" << std::endl;
}

void test_broadcast_insert2_ratios_cross_validate() {
  std::cerr << "=== test_broadcast_insert2_ratios_cross_validate ===" << std::endl;
  fun f;
  triqs::det_manip::det_manip<fun> D(f, 100);
  triqs::det_manip::det_manip_basic<fun> Db(f, 100);
  triqs::mc_tools::random_generator RNG("mt19937", 88990);

  for (int n = 0; n < 15; ++n) {
    double x = RNG(10.0);
    double y = RNG(10.0);
    D.insert(D.size(), D.size(), x, y);
    Db.insert(Db.size(), Db.size(), x, y);
  }

  long M = 4, E = 6;
  auto x0s = random_array2(M, E, RNG);
  auto y0s = random_array2(M, E, RNG);
  auto x1s = random_array1(E, RNG);
  auto y1s = random_array1(E, RNG);

  D.regenerate(); // refresh inverse for numerical accuracy before cross-validation

  auto batch_opt   = D.insert2_ratios(0, 1, 0, 1, x0s, x1s, y0s, y1s);
  auto batch_basic = Db.insert2_ratios(0, 1, 0, 1, x0s, x1s, y0s, y1s);

  for (long i = 0; i < M; ++i)
    for (long j = 0; j < E; ++j)
      assert_close(batch_opt(i, j), batch_basic(i, j), 1.e-2,
                   "cross broadcast insert2 (" + std::to_string(i) + "," + std::to_string(j) + ")");
  std::cerr << "PASSED" << std::endl;
}

void test_broadcast_insert2_ratios_reversed() {
  std::cerr << "=== test_broadcast_insert2_ratios_reversed ===" << std::endl;
  fun f;
  triqs::det_manip::det_manip<fun> D(f, 100);
  triqs::mc_tools::random_generator RNG("mt19937", 99001);
  build_det(D, 20, RNG);

  long M = 4, E = 6;
  // Pair 0 (A-side): rank-1, shape (E) -- broadcast
  auto x0s = random_array1(E, RNG);
  auto y0s = random_array1(E, RNG);
  // Pair 1 (B-side): rank-2, shape (M, E) -- high-rank
  auto x1s = random_array2(M, E, RNG);
  auto y1s = random_array2(M, E, RNG);

  auto batch = D.insert2_ratios(0, 1, 0, 1, x0s, x1s, y0s, y1s);
  static_assert(nda::get_rank<decltype(batch)> == 2);
  TRIQS_ASSERT(batch.shape() == (std::array<long, 2>{M, E}));

  for (long i = 0; i < M; ++i)
    for (long j = 0; j < E; ++j) {
      auto ratio = D.try_insert2(0, 1, 0, 1, x0s(j), x1s(i, j), y0s(j), y1s(i, j));
      D.reject_last_try();
      assert_close(batch(i, j), ratio, PRECISION,
                   "broadcast reversed insert2 (" + std::to_string(i) + "," + std::to_string(j) + ")");
    }
  std::cerr << "PASSED" << std::endl;
}

void test_broadcast_insert2_empty_matrix() {
  std::cerr << "=== test_broadcast_insert2_empty_matrix ===" << std::endl;
  fun f;
  triqs::det_manip::det_manip<fun> D(f, 100);

  long M = 3, E = 4;
  triqs::mc_tools::random_generator RNG("mt19937", 10101);
  auto x0s = random_array2(M, E, RNG);
  auto y0s = random_array2(M, E, RNG);
  auto x1s = random_array1(E, RNG);
  auto y1s = random_array1(E, RNG);

  auto batch = D.insert2_ratios(0, 1, 0, 1, x0s, x1s, y0s, y1s);

  for (long i = 0; i < M; ++i)
    for (long j = 0; j < E; ++j) {
      auto ratio = D.try_insert2(0, 1, 0, 1, x0s(i, j), x1s(j), y0s(i, j), y1s(j));
      D.reject_last_try();
      assert_close(batch(i, j), ratio, 1.e-14,
                   "broadcast empty insert2 (" + std::to_string(i) + "," + std::to_string(j) + ")");
    }
  std::cerr << "PASSED" << std::endl;
}

int main() {
  // Original rank-1 tests
  test_rank1_batch_vs_sequential();
  test_rank1_state_unchanged();
  test_rank1_empty_matrix();
  test_rank1_single_point();
  test_rank1_empty_batch();
  test_rank2_batch_vs_sequential();
  test_rank2_state_unchanged();
  test_rank2_empty_matrix();
  test_rank2_reversed_indices();
  test_cross_validate_rank1();
  test_cross_validate_rank2();
  test_matrix_vs_sequential();
  test_matrix_state_unchanged();
  test_matrix_empty();
  test_matrix_cross_validate();
  test_matrix_vs_paired();

  // Rank-2 array tests
  test_rank2_array_insert_ratios();
  test_rank2_array_insert_ratios_cross_validate();
  test_rank2_array_insert2_ratios();

  // Broadcast tests (mixed rank-2 / rank-1)
  test_broadcast_insert2_ratios();
  test_broadcast_insert2_ratios_basic_vs_sequential();
  test_broadcast_insert2_ratios_cross_validate();
  test_broadcast_insert2_ratios_reversed();
  test_broadcast_insert2_empty_matrix();

  std::cerr << "\nAll tests PASSED." << std::endl;
  return 0;
}
