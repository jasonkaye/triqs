// Copyright (c) 2013-2018 Commissariat à l'énergie atomique et aux énergies alternatives (CEA)
// Copyright (c) 2013-2018 Centre national de la recherche scientifique (CNRS)
// Copyright (c) 2018-2020 Simons Foundation
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

/**
 * @file
 * @brief Provides a type erased random number generator based on 64-bit standard library engines.
 */

#pragma once

#include <h5/h5.hpp>

#include <cassert>
#include <concepts>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <sstream>
#include <utility>
#include <vector>

namespace triqs::mc_tools {

  /**
   * @addtogroup triqs-mc-utils
   * @{
   */

  /**
   * @brief Wrapper that erases the type of a random number generator.
   *
   * @details All supported engines produce 64-bit unsigned integers internally via the standard library.
   * Engines with native output smaller than 64 bits are wrapped using `std::independent_bits_engine`.
   *
   * The following engines are supported (see also triqs::mc_tools::random_generator_names_list()):
   *
   * - *empty string* or *mt19937_64*: uses `std::mt19937_64` (default, native 64-bit)
   * - *mt19937*: uses `std::mt19937` (32-bit, combined to 64-bit)
   * - *ranlux48*: uses `std::ranlux48` (48-bit, combined to 64-bit)
   * - *ranlux24*: uses `std::ranlux24` (24-bit, combined to 64-bit)
   * - *minstd_rand*: uses `std::minstd_rand` (31-bit, combined to 64-bit)
   * - *knuth_b*: uses `std::knuth_b` (31-bit, combined to 64-bit)
   *
   * For performance, raw `uint64_t` values are generated in batches and stored in a buffer.
   * Doubles in [0, 1) are derived using the standard 53-bit technique.
   * Integers in [0, i) are generated using Lemire's nearly divisionless method (unbiased for all ranges).
   */
  class random_generator {
    private:
    // RNG concept defines the interface for RNGs producing uint64_t.
    struct rng_concept {
      virtual ~rng_concept()                                 = default;
      virtual std::uint64_t operator()()                     = 0;
      virtual void refill(std::vector<std::uint64_t> &)      = 0;
      virtual std::ostream &to_ostream(std::ostream &) const = 0;
      virtual std::istream &from_istream(std::istream &)     = 0;
    };

    // RNG model wraps a concrete engine that produces uint64_t.
    template <typename T> struct rng_model : public rng_concept {
      T engine_;
      rng_model(T engine) : engine_{std::move(engine)} {}
      std::uint64_t operator()() override { return engine_(); }
      void refill(std::vector<std::uint64_t> &buffer) override {
        for (auto &x : buffer) x = engine_();
      }
      std::ostream &to_ostream(std::ostream &os) const override {
        os << engine_;
        return os;
      }
      std::istream &from_istream(std::istream &is) override {
        is >> engine_;
        return is;
      }
    };

    public:
    /// Default seed for the underlying RNG.
    static constexpr std::uint64_t default_seed = 198;

    /// Default constructor uses std::mt19937_64 RNG.
    random_generator() : random_generator("mt19937_64", default_seed) {}

    /**
     * @brief Construct a random generator by wrapping the specified RNG and seeding it with the given seed.
     *
     * @details The given name has to correspond to one of the supported RNGs (see
     * triqs::mc_tools::random_generator_names() or triqs::mc_tools::random_generator_names_list()). If the name does
     * not match any of the supported RNGs, a `std::runtime_error` is thrown.
     *
     * An empty name corresponds to the default RNG (std::mt19937_64).
     *
     * @param name Name of the RNG to be used.
     * @param seed Seed for the RNG.
     * @param buffer_size Size of the buffer used to store random numbers.
     */
    random_generator(std::string name, std::uint64_t seed, std::size_t buffer_size = 1000);

    /// Deleted copy constructor.
    random_generator(random_generator const &) = delete;

    /// Default move constructor.
    random_generator(random_generator &&) = default;

    /// Default move assignment operator.
    random_generator &operator=(random_generator &&) = default;

    /// Get the name of the underlying RNG.
    [[nodiscard]] std::string name() const { return name_; }

    /**
     * @brief Generate a random sample from the uniform integer distribution defined on \f$ \{0, ..., i-1 \}\f$.
     *
     * @details Uses Lemire's nearly divisionless method for unbiased generation across all ranges,
     * including very large ranges up to UINT64_MAX.
     *
     * @tparam T Integral type.
     * @param i Upper bound (excluded).
     * @return Uniform random integer in [0, i).
     */
    template <typename T>
      requires(std::integral<T>)
    T operator()(T i) {
      if (i <= 1) return 0;
      auto range = static_cast<std::uint64_t>(i);
      auto x     = raw_uint64();
      // Lemire's nearly divisionless method
      __uint128_t m = __uint128_t(x) * __uint128_t(range);
      auto l        = static_cast<std::uint64_t>(m);
      if (l < range) {
        auto t = -range % range; // rejection threshold
        while (l < t) {
          x = raw_uint64();
          m = __uint128_t(x) * __uint128_t(range);
          l = static_cast<std::uint64_t>(m);
        }
      }
      return static_cast<T>(m >> 64);
    }

    /**
     * @brief Look ahead at the next value that will be generated with a call to operator()().
     * @return Uniform random double from the interval \f$ [0, 1) \f$.
     */
    [[nodiscard]] double preview() {
      if (idx_ > buffer_.size() - 1) refill();
      return to_double(buffer_[idx_]);
    }

    /**
     * @brief Generate a random sample from the uniform distribution defined on the interval \f$ [0, 1) \f$.
     *
     * @details Uses the standard 53-bit technique: the upper 53 bits of a 64-bit integer are scaled
     * to produce a double with full mantissa precision.
     *
     * @return Uniform random double from the interval \f$ [0, 1) \f$.
     */
    double operator()() { return to_double(raw_uint64()); }

    /**
     * @brief Generate a random sample from the uniform distribution defined on the interval \f$ [0, b) \f$.
     * @param b Upper bound of the interval.
     * @return Uniform random double from the interval \f$ [0, b) \f$.
     */
    double operator()(double b) { return b * (this->operator()()); }

    /**
     * @brief Generate a random sample from the uniform distribution defined on the interval \f$ [a, b) \f$.
     *
     * @param a Lower bound of the interval.
     * @param b Upper bound of the interval.
     * @return Uniform random double from the interval \f$ [a, b) \f$.
     */
    double operator()(double a, double b) {
      assert(b > a);
      return a + (b - a) * (this->operator()());
    }

    /// Get the HDF5 format tag.
    [[nodiscard]] static std::string hdf5_format() { return "random_generator"; }

    /**
     * @brief Write the RNG object to HDF5.
     *
     * @param g `h5::group` to be written to.
     * @param name Name of the dataset/subgroup.
     * @param rng RNG object to be written.
     */
    friend void h5_write(h5::group g, std::string const &name, random_generator const &rng) {
      auto gr = g.create_group(name);
      h5::write_hdf5_format(gr, rng); // NOLINT (downcasting to base class)
      h5::write(gr, "name", rng.name_);
      h5::write(gr, "buffer", rng.buffer_);
      h5::write(gr, "idx", rng.idx_);
      std::ostringstream os;
      rng.ptr_->to_ostream(os);
      h5::write(gr, "rng", os.str());
    }

    /**
     * @brief Read the RNG object from HDF5.
     *
     * @param g `h5::group` to be read from.
     * @param name Name of the dataset/subgroup.
     * @param rng RNG object to be read into.
     */
    friend void h5_read(h5::group g, std::string const &name, random_generator &rng) {
      auto gr = g.open_group(name);
      h5::assert_hdf5_format(gr, rng); // NOLINT (downcasting to base class)
      h5::read(gr, "name", rng.name_);
      h5::read(gr, "buffer", rng.buffer_);
      h5::read(gr, "idx", rng.idx_);
      rng.initialize_rng(rng.name_, default_seed);
      std::string rng_state;
      h5::read(gr, "rng", rng_state);
      std::istringstream is{rng_state};
      rng.ptr_->from_istream(is);
    }

    private:
    // Convert a raw uint64_t to a double in [0, 1) using the 53-bit technique.
    static double to_double(std::uint64_t x) { return (x >> 11) * 0x1.0p-53; }

    // Get the next raw uint64_t from the buffer.
    std::uint64_t raw_uint64() {
      if (idx_ > buffer_.size() - 1) refill();
      return buffer_[idx_++];
    }

    // Refill the buffer.
    void refill() {
      ptr_->refill(buffer_);
      idx_ = 0;
    }

    // Initialize the RNG.
    void initialize_rng(std::string const &name, std::uint64_t seed);

    private:
    std::unique_ptr<rng_concept> ptr_;
    size_t idx_{0};
    std::vector<std::uint64_t> buffer_;
    std::string name_;
  };

  /**
   * @brief Get a string containing the names of all available RNGs.
   * @param sep Separator between the names.
   * @return `std::string` containing the available RNGs separated by the given separator.
   */
  [[nodiscard]] std::string random_generator_names(std::string const &sep = " ");

  /// Get a `std::vector<std::string>` containing all available RNG names.
  [[nodiscard]] std::vector<std::string> random_generator_names_list();

  /** @} */

} // namespace triqs::mc_tools
