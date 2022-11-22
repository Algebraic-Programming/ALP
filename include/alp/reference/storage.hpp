/*
 *   Copyright 2021 Huawei Technologies Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 *
 * @file
 *
 * This file registers mechanisms for coordinate mapping between
 * logical and physical iteration spaces.
 *
 */

#ifndef _H_ALP_REFERENCE_STORAGE
#define _H_ALP_REFERENCE_STORAGE

#include <alp/amf-based/storage.hpp>

namespace alp {

	namespace internal {

		/** Specialization for general matrix */
		template<>
		struct determine_poly_factory< structures::General, imf::Id, imf::Id, reference > {

			typedef storage::polynomials::FullFactory<> factory_type;
		};

		/** Specialization for square matrix */
		template<>
		struct determine_poly_factory< structures::Square, imf::Id, imf::Id, reference > {

			typedef storage::polynomials::FullFactory<> factory_type;
		};

		/** Specialization for orthogonal matrix */
		template<>
		struct determine_poly_factory< structures::Orthogonal, imf::Id, imf::Id, reference > {

			typedef storage::polynomials::FullFactory<> factory_type;
		};

		/** Specialization for upper-triangular matrix */
		template<>
		struct determine_poly_factory< structures::UpperTriangular, imf::Id, imf::Id, reference > {

			typedef storage::polynomials::PackedFactory< storage::UPPER, storage::ROW_WISE > factory_type;
		};

		/** Specialization for lower-triangular matrix */
		template<>
		struct determine_poly_factory< structures::LowerTriangular, imf::Id, imf::Id, reference > {

			typedef storage::polynomials::PackedFactory< storage::LOWER, storage::ROW_WISE > factory_type;
		};

		/** Specialization for upper-trapezoidal matrix */
		template<>
		struct determine_poly_factory< structures::UpperTrapezoidal, imf::Id, imf::Id, reference > {

			typedef storage::polynomials::FullFactory<> factory_type;
		};

		/** Specialization for lower-trapezoidal matrix */
		template<>
		struct determine_poly_factory< structures::LowerTrapezoidal, imf::Id, imf::Id, reference > {

			typedef storage::polynomials::FullFactory<> factory_type;
		};

		/** Specialization for symmetric matrix */
		template<>
		struct determine_poly_factory< structures::Symmetric, imf::Id, imf::Id, reference > {

			typedef storage::polynomials::PackedFactory< storage::UPPER, storage::ROW_WISE > factory_type;
		};

		/** Specialization for hermitian matrix */
		template<>
		struct determine_poly_factory< structures::Hermitian, imf::Id, imf::Id, reference > {

			typedef storage::polynomials::FullFactory<> factory_type;
		};

		/** Specialization for symmetric positive definite matrix */
		template<>
		struct determine_poly_factory< structures::SymmetricPositiveDefinite, imf::Id, imf::Id, reference > {

			typedef storage::polynomials::PackedFactory< storage::UPPER, storage::ROW_WISE > factory_type;
		};

		/** Specialization for hermitian positive definite matrix */
		template<>
		struct determine_poly_factory< structures::HermitianPositiveDefinite, imf::Id, imf::Id, reference > {

			typedef storage::polynomials::FullFactory<> factory_type;
		};

		/** Specialization for symmetric tridiagonal matrix */
		template<>
		struct determine_poly_factory< structures::SymmetricTridiagonal, imf::Id, imf::Id, reference > {

			private:
				using interval = std::tuple_element< 0, structures::SymmetricTridiagonal::band_intervals >::type;

			public:
				//typedef storage::polynomials::BandFactory< interval, storage::ROW_WISE > factory_type;
				typedef storage::polynomials::FullFactory<> factory_type;
		};

		/** Specialization for hermitian tridiagonal matrix */
		template<>
		struct determine_poly_factory< structures::HermitianTridiagonal, imf::Id, imf::Id, reference > {

			private:
				// This will be used in the commented line below once band storage is added.
				// Added for readability.
				using interval = std::tuple_element< 0, structures::SymmetricTridiagonal::band_intervals >::type;

			public:
				//typedef storage::polynomials::BandFactory< interval, storage::ROW_WISE > factory_type;
				typedef storage::polynomials::FullFactory<> factory_type;
		};

		/** Specialization for rectangular-upper-bidiagonal matrix */
		template<>
		struct determine_poly_factory< structures::RectangularUpperBidiagonal, imf::Id, imf::Id, reference > {
			// should use band storage
			typedef storage::polynomials::FullFactory<> factory_type;
		};

		/** Specialization for vectors */
		template< typename Structure >
		struct determine_poly_factory< Structure, imf::Id, imf::Zero, reference > {

			typedef storage::polynomials::ArrayFactory factory_type;
		};

	} // namespace internal

} // namespace alp

#endif // _H_ALP_REFERENCE_STORAGE
