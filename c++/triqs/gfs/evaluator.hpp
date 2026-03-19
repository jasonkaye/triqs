// Copyright (c) 2016-2018 Commissariat à l'énergie atomique et aux énergies alternatives (CEA)
// Copyright (c) 2016-2018 Centre national de la recherche scientifique (CNRS)
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

#pragma once

namespace triqs::gfs {

  // Default evaluation: forward to mesh-level evaluate with zero check
  namespace detail {
    template <typename G, typename... XS>
    auto eval_default(G const &g, XS &&...xs) {
      auto l    = [&g](auto &&...ys) -> decltype(auto) { return g.operator[](ys...); };
      using r_t = std::decay_t<decltype(make_regular(evaluate(g.mesh(), l, std::forward<XS>(xs)...)))>;
      if constexpr (nda::Array<r_t> or nda::is_scalar_v<r_t>) {
        if (eval_to_zero(g.mesh(), xs...)) { return r_t{nda::zeros<typename G::target_t::scalar_t>(g.target_shape())}; }
      }
      return make_regular(evaluate(g.mesh(), l, std::forward<XS>(xs)...));
    }
  } // namespace detail

  /*----------------------------------------------------------
  *  Default
  *--------------------------------------------------------*/

  template <Mesh M> struct gf_evaluator {

    template <typename G, typename... XS>
      requires(is_gf_v<G>)
    auto operator()(G const &g, XS &&...xs) const {
      return detail::eval_default(g, std::forward<XS>(xs)...);
    }
  };

  /*----------------------------------------------------------
   *  mesh::prod — Optimized evaluator for GFs on product
   *  meshes. For scalar-valued targets, resolves leading
   *  discrete dimensions (index_t or mesh_point_t arguments)
   *  via operator[], avoiding nested currying overhead. When
   *  all dimensions are discrete, uses direct pointer arithmetic.
   *--------------------------------------------------------*/

  namespace detail {

    // True if argument type Xk can be resolved to a data index for mesh Mk
    template <typename Mk, typename Xk>
    constexpr bool is_resolvable_v = !std::is_same_v<Xk, nda::range::all_t>
       && (std::is_same_v<Xk, typename Mk::index_t> || std::is_same_v<Xk, typename Mk::mesh_point_t>);

    // Count leading resolvable dimensions at compile time
    template <typename MsTuple, typename XsTuple, size_t K = 0>
    constexpr size_t count_leading_resolvable() {
      if constexpr (K >= std::tuple_size_v<MsTuple>)
        return K;
      else if constexpr (is_resolvable_v<std::tuple_element_t<K, MsTuple>, std::tuple_element_t<K, XsTuple>>)
        return count_leading_resolvable<MsTuple, XsTuple, K + 1>();
      else
        return K;
    }

    // Resolve a single argument to a data index
    template <typename Mk, typename Xk>
    FORCEINLINE long to_data_idx(Mk const &mk, Xk const &xk) {
      if constexpr (std::is_same_v<Xk, typename Mk::mesh_point_t>)
        return xk.data_index();
      else
        return mk.to_data_index(xk);
    }

    // Accumulate pointer offset for dimensions [From, To)
    template <size_t From, size_t To, typename G, typename ArgsTuple>
    FORCEINLINE long compute_offset(G const &g, ArgsTuple const &args) {
      return [&]<size_t... Is>(std::index_sequence<Is...>) {
        return ((to_data_idx<std::tuple_element_t<From + Is, typename std::decay_t<decltype(g.mesh())>::m_tuple_t>>(
                    std::get<From + Is>(g.mesh()), std::get<From + Is>(args))
                 * g.data().indexmap().strides()[From + Is])
                + ...);
      }(std::make_index_sequence<To - From>{});
    }

    // Extract elements [From, To) from a tuple as a tuple of const references
    template <size_t From, size_t To, typename Tuple>
    auto sub_tuple(Tuple const &t) {
      return [&]<size_t... Is>(std::index_sequence<Is...>) {
        return std::tie(std::get<From + Is>(t)...);
      }(std::make_index_sequence<To - From>{});
    }

  } // namespace detail

  template <Mesh... Ms> struct gf_evaluator<mesh::prod<Ms...>> {

    template <typename G, typename... XS>
      requires(is_gf_v<G>)
    auto operator()(G const &g, XS &&...xs) const {
      constexpr size_t N        = sizeof...(Ms);
      constexpr bool has_all_t  = (std::is_same_v<std::decay_t<XS>, nda::range::all_t> or ...);

      if constexpr (N == sizeof...(XS) and G::target_t::rank == 0 and not has_all_t) {
        // Return zero if any mesh component evaluates to zero (e.g. out-of-range Matsubara frequency)
        if (detail::eval_to_zero(g.mesh(), xs...)) return typename G::target_t::scalar_t{0};
        constexpr size_t K = detail::count_leading_resolvable<std::tuple<Ms...>, std::tuple<std::decay_t<XS>...>>();

        if constexpr (K == N) {
          // All dimensions discrete: direct data access
          auto args = std::forward_as_tuple(xs...);
          return *(g.data().data() + detail::compute_offset<0, N>(g, args));

        } else if constexpr (K > 0) {
          // Resolve leading discrete args via operator[], delegate remaining to evaluate
          auto args    = std::forward_as_tuple(xs...);
          auto leading = detail::sub_tuple<0, K>(args);
          auto f       = [&g, &leading](auto const &...ys) {
            return std::apply([&](auto const &...ls) { return g[ls..., ys...]; }, leading);
          };
          return std::apply([&](auto const &...rem_xs) { return evaluate(detail::sub_tuple<K, N>(g.mesh().components()), f, rem_xs...); },
                            detail::sub_tuple<K, N>(args));

        } else {
          return detail::eval_default(g, std::forward<XS>(xs)...);
        }
      } else {
        return detail::eval_default(g, std::forward<XS>(xs)...);
      }
    }
  };

  /*----------------------------------------------------------
   *  mesh::imfreq
   *--------------------------------------------------------*/

  template <> struct gf_evaluator<mesh::imfreq> {

    template <typename G> auto operator()(G const &g, matsubara_freq const &f) const {

      using r_t = std::decay_t<decltype(make_regular(g[0]))>;

      if (g.mesh().is_index_valid(f.n)) return r_t{g[f.n]};
      if (g.mesh().positive_only()) {
        int sh = (g.mesh().statistic() == Fermion ? 1 : 0);
        if (g.mesh().is_index_valid(-f.n - sh)) return r_t{conj(g[-f.n - sh])};
        TRIQS_RUNTIME_ERROR << " ERROR: Cannot evaluate Green function with positive only mesh outside grid ";
      }

      auto [tail, err] = fit_tail_no_normalize(g);
      dcomplex x       = std::abs(g.mesh().w_max()) / f;
      auto res         = r_t{nda::zeros<dcomplex>(g.target_shape())}; // a new array

      dcomplex z = 1.0;
      for (int n : range(tail.extent(0))) {
        res += tail(n, ellipsis()) * z;
        z = z * x;
      }

      return res;
    }

    // int -> replace by matsubara_freq
    template <typename G> decltype(auto) operator()(G const &g, int n) const { return g(matsubara_freq(n, g.mesh().beta(), g.mesh().statistic())); }
  };

  /*----------------------------------------------------------
   *  mesh::dlr2d
   *--------------------------------------------------------*/

  template <> struct gf_evaluator<mesh::dlr2d> {

    // Handle pair of matsubara_freq: G({iw1, iw2})
    template <typename G> auto operator()(G const &g, std::pair<matsubara_freq, matsubara_freq> const &iw_pair) const {
      auto l = [&g](auto i) -> decltype(auto) { return g[i]; };
      return make_regular(evaluate(g.mesh(), l, iw_pair));
    }
  };

} // namespace triqs::gfs
