// Copyright (c) 2013-2018 Commissariat à l'énergie atomique et aux énergies alternatives (CEA)
// Copyright (c) 2013-2018 Centre national de la recherche scientifique (CNRS)
// Copyright (c) 2018-2023 Simons Foundation
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
 * @brief Implementation details for triqs/mc_tools/random_generator.hpp.
 */

#include "./random_generator.hpp"
#include "../utility/first_include.hpp"

#include <fmt/format.h>

#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace triqs::mc_tools {

  // List of supported engine names (excluding empty string which maps to mt19937_64).
  static const std::vector<std::string> engine_names = {"mt19937_64", "mt19937", "ranlux48", "ranlux24", "minstd_rand", "knuth_b"};

  random_generator::random_generator(std::string name, std::uint64_t seed, std::size_t buffer_size) : buffer_(buffer_size), name_(std::move(name)) {
    initialize_rng(name_, seed);
    refill();
  }

  void random_generator::initialize_rng(std::string const &name, std::uint64_t seed) {

    // mt19937_64: native 64-bit engine (default)
    if (name.empty() || name == "mt19937_64") {
      ptr_ = std::make_unique<rng_model<std::mt19937_64>>(std::mt19937_64{seed});
      return;
    }

    // mt19937: 32-bit engine, combined to 64-bit via independent_bits_engine
    if (name == "mt19937") {
      using engine_t = std::independent_bits_engine<std::mt19937, 64, std::uint64_t>;
      ptr_           = std::make_unique<rng_model<engine_t>>(engine_t{static_cast<std::uint32_t>(seed)});
      return;
    }

    // ranlux48: 48-bit engine, combined to 64-bit
    if (name == "ranlux48") {
      using engine_t = std::independent_bits_engine<std::ranlux48, 64, std::uint64_t>;
      ptr_           = std::make_unique<rng_model<engine_t>>(engine_t{static_cast<std::uint32_t>(seed)});
      return;
    }

    // ranlux24: 24-bit engine, combined to 64-bit
    if (name == "ranlux24") {
      using engine_t = std::independent_bits_engine<std::ranlux24, 64, std::uint64_t>;
      ptr_           = std::make_unique<rng_model<engine_t>>(engine_t{static_cast<std::uint32_t>(seed)});
      return;
    }

    // minstd_rand: 31-bit LCG, combined to 64-bit
    if (name == "minstd_rand") {
      using engine_t = std::independent_bits_engine<std::minstd_rand, 64, std::uint64_t>;
      ptr_           = std::make_unique<rng_model<engine_t>>(engine_t{static_cast<std::uint32_t>(seed)});
      return;
    }

    // knuth_b: 31-bit shuffle engine, combined to 64-bit
    if (name == "knuth_b") {
      using engine_t = std::independent_bits_engine<std::knuth_b, 64, std::uint64_t>;
      ptr_           = std::make_unique<rng_model<engine_t>>(engine_t{static_cast<std::uint32_t>(seed)});
      return;
    }

    throw std::runtime_error(fmt::format("Error in random_generator::initialize_rng: RNG with name '{}' is not supported", name));
  }

  std::string random_generator_names(std::string const &sep) {
    std::string result;
    for (std::size_t i = 0; i < engine_names.size(); ++i) {
      if (i > 0) result += sep;
      result += engine_names[i];
    }
    return result;
  }

  std::vector<std::string> random_generator_names_list() { return engine_names; }

} // namespace triqs::mc_tools
