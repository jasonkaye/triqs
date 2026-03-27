// Copyright (c) 2015-2018 Commissariat à l'énergie atomique et aux énergies alternatives (CEA)
// Copyright (c) 2015-2018 Centre national de la recherche scientifique (CNRS)
// Copyright (c) 2018 Simons Foundation
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
//
// Authors: Michel Ferrero, Olivier Parcollet, Nils Wentzell

#include <triqs/mc_tools/random_generator.hpp>
#include <triqs/test_tools/arrays.hpp>

#include <fmt/ranges.h>

#include <cstdint>
#include <random>
#include <string>
#include <vector>

TEST(TRIQSMCTools, RandomGeneratorNames) {
  fmt::print("{}\n", triqs::mc_tools::random_generator_names());
  auto names = triqs::mc_tools::random_generator_names_list();
  fmt::print("{}\n", names);
  EXPECT_EQ(names.size(), 6);
  EXPECT_EQ(names[0], "mt19937_64");
}

// Verify that the default (empty string) and "mt19937_64" produce the same sequence.
TEST(TRIQSMCTools, DefaultIsMT19937_64) {
  using namespace triqs::mc_tools;
  int const seed = 42;
  auto rng_default = random_generator("", seed);
  auto rng_named   = random_generator("mt19937_64", seed);
  for (int i = 0; i < 100; ++i) EXPECT_DOUBLE_EQ(rng_default(), rng_named());
}

// Verify that the wrapped mt19937_64 produces the same double sequence as direct usage
// with our 53-bit conversion formula.
TEST(TRIQSMCTools, MT19937_64MatchesDirectUsage) {
  using namespace triqs::mc_tools;
  std::uint64_t const seed = 0x18a2b3c4;
  auto rng                 = random_generator("mt19937_64", seed);
  std::mt19937_64 direct_engine{seed};

  for (int i = 0; i < 100; ++i) {
    double const from_rng    = rng();
    auto raw                 = direct_engine();
    double const from_direct = (raw >> 11) * 0x1.0p-53;
    EXPECT_DOUBLE_EQ(from_rng, from_direct);
  }
}

// Verify seed reproducibility: same engine + same seed = same sequence.
TEST(TRIQSMCTools, SeedReproducibility) {
  using namespace triqs::mc_tools;
  for (auto const &name : random_generator_names_list()) {
    auto rng1 = random_generator(name, 12345);
    auto rng2 = random_generator(name, 12345);
    for (int i = 0; i < 50; ++i) EXPECT_DOUBLE_EQ(rng1(), rng2()) << "Engine: " << name;
  }
}

// Verify all engines produce valid doubles in [0, 1).
TEST(TRIQSMCTools, AllEnginesDoubleRange) {
  using namespace triqs::mc_tools;
  for (auto const &name : random_generator_names_list()) {
    auto rng = random_generator(name, 54321);
    for (int i = 0; i < 10000; ++i) {
      double val = rng();
      EXPECT_GE(val, 0.0) << "Engine: " << name;
      EXPECT_LT(val, 1.0) << "Engine: " << name;
    }
  }
}

// Verify all engines produce valid integers in the requested range.
TEST(TRIQSMCTools, AllEnginesIntegerRange) {
  using namespace triqs::mc_tools;
  for (auto const &name : random_generator_names_list()) {
    auto rng = random_generator(name, 99999);
    for (int i = 0; i < 10000; ++i) {
      auto val = rng(100);
      EXPECT_GE(val, 0);
      EXPECT_LT(val, 100);
    }
  }
}

// Integer generation at large ranges: verify results span the full 64-bit range.
TEST(TRIQSMCTools, LargeRangeIntegerGeneration) {
  using namespace triqs::mc_tools;
  auto rng                  = random_generator("mt19937_64", 42);
  constexpr auto range      = std::numeric_limits<std::uint64_t>::max();
  bool has_high_bits        = false;
  bool has_low_bits         = false;
  constexpr auto half_range = range / 2;

  for (int i = 0; i < 10000; ++i) {
    auto val = rng(range);
    if (val > half_range) has_high_bits = true;
    if (val < half_range) has_low_bits = true;
    EXPECT_LT(val, range);
  }
  EXPECT_TRUE(has_high_bits) << "No values in upper half of 64-bit range";
  EXPECT_TRUE(has_low_bits) << "No values in lower half of 64-bit range";
}

// Integer uniformity: chi-squared test for moderate range.
TEST(TRIQSMCTools, IntegerUniformity) {
  using namespace triqs::mc_tools;
  auto rng           = random_generator("mt19937_64", 42);
  constexpr int bins = 100;
  constexpr int N    = 1000000;
  std::vector<int> counts(bins, 0);

  for (int i = 0; i < N; ++i) { ++counts[rng(bins)]; }

  double expected = static_cast<double>(N) / bins;
  double chi2     = 0.0;
  for (int c : counts) { chi2 += (c - expected) * (c - expected) / expected; }

  // chi-squared with 99 df: p=0.001 critical value is ~148.2
  EXPECT_LT(chi2, 150.0) << "Chi-squared test failed: distribution is not uniform";
}

// Double generation quality: verify 53 bits of mantissa are used.
TEST(TRIQSMCTools, DoublePrecision53Bits) {
  using namespace triqs::mc_tools;
  auto rng = random_generator("mt19937_64", 42);

  // Count distinct values in a small interval. With 53-bit precision,
  // values near 0.5 should have spacing ~2^-53 ≈ 1.1e-16.
  // With only 32-bit precision, spacing would be ~2^-32 ≈ 2.3e-10.
  int distinct_low_bits = 0;
  for (int i = 0; i < 100000; ++i) {
    double val = rng();
    // Check if the value has non-trivial bits below the 32-bit precision threshold.
    // Multiply by 2^53 and check if the lower 21 bits (53-32) are non-zero.
    auto scaled   = static_cast<std::uint64_t>(val * (1ULL << 53));
    auto low_bits = scaled & ((1ULL << 21) - 1);
    if (low_bits != 0) ++distinct_low_bits;
  }

  // With true 53-bit precision, roughly half the values should have non-zero lower 21 bits.
  EXPECT_GT(distinct_low_bits, 40000) << "Doubles do not appear to use full 53-bit precision";
}

// Preview returns the same value as the next call to operator().
TEST(TRIQSMCTools, PreviewConsistency) {
  using namespace triqs::mc_tools;
  auto rng = random_generator("mt19937_64", 42);
  for (int i = 0; i < 100; ++i) {
    double preview_val = rng.preview();
    double actual_val  = rng();
    EXPECT_DOUBLE_EQ(preview_val, actual_val);
  }
}

// HDF5 round-trip: save/restore and verify continued sequence.
TEST(TRIQSMCTools, RandomGeneratorHDF5) {
  using namespace triqs::mc_tools;
  auto check_hdf5 = [](std::string const &name) {
    auto rng = random_generator(name, 0x18a2b3c4);
    for (int i = 0; i < 10; ++i) rng();
    auto rng2 = rw_h5(rng, "mctools_rng_" + (name.empty() ? "default" : name), name.empty() ? "default" : name);
    for (int i = 0; i < 10; ++i) EXPECT_DOUBLE_EQ(rng(), rng2()) << "HDF5 round-trip failed for engine: " << name;
  };

  for (auto const &name : random_generator_names_list()) check_hdf5(name);
  check_hdf5(""); // default
}

// Move semantics: verify move constructor and assignment produce identical sequences.
TEST(TRIQSMCTools, RandomGeneratorMoveOperation) {
  using namespace triqs::mc_tools;
  auto rng  = random_generator();
  auto rng2 = random_generator();
  for (int i = 0; i < 10; ++i) {
    rng();
    rng2();
  }

  // move constructor
  auto rng3 = std::move(rng);
  for (int i = 0; i < 10; ++i) EXPECT_DOUBLE_EQ(rng2(), rng3());

  // move assignment
  auto rng4 = random_generator();
  rng4      = std::move(rng2);
  for (int i = 0; i < 10; ++i) EXPECT_DOUBLE_EQ(rng3(), rng4());
}

MAKE_MAIN;
