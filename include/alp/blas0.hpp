
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

/*
 * @author A. N. Yzelman
 * @date 5th of December 2016
 */

#ifndef _H_ALP_BLAS0
#define _H_ALP_BLAS0

#include "base/blas0.hpp"
#include "descriptors.hpp"

// now include all specialisations contained in the backend directories:
#ifdef _ALP_WITH_REFERENCE
 #include <alp/reference/blas0.hpp>
#endif


namespace alp {

	namespace internal {

		/**
		 * Helper class that, depending on a given descriptor, either returns a
		 * nonzero value from a vector, or its corresponding coordinate.
		 *
		 * This class hence makes the use of the following descriptor(s) transparent:
		 *   -# #alp::descriptors::use_index
		 *
		 * @tparam descr The descriptor under which to write back either the value or
		 *               the index.
		 * @tparam OutputType The type of the output to return.
		 * @tparam D          The type of the input.
		 * @tparam Enabled    Controls, through SFINAE, whether the use of the
		 *                    #use_index descriptor is allowed at all.
		 */
		template< alp::Descriptor descr, typename OutputType, typename D, typename Enabled = void >
		class ValueOrIndex;

		/* Version where use_index is allowed. */
		template< alp::Descriptor descr, typename OutputType, typename D >
		class ValueOrIndex< 
			descr, OutputType, D,
			typename std::enable_if< std::is_arithmetic< OutputType >::value
			&& ! std::is_same< D, void >::value >::type 
		> {
		private:
			static constexpr const bool use_index = descr & alp::descriptors::use_index;
			static_assert(
				use_index
				|| std::is_convertible< D, OutputType >::value, "Cannot convert to the requested output type"
			);

		public:

			static OutputType getFromScalar( const D &x, const size_t index ) noexcept {
				if( use_index ) {
					return static_cast< OutputType >( index );
				} else {
					return static_cast< OutputType >( x );
				}
			}

		};

		/* Version where use_index is not allowed. */
		template< alp::Descriptor descr, typename OutputType, typename D >
		class ValueOrIndex<
			descr, OutputType, D,
			typename std::enable_if< ! std::is_arithmetic< OutputType >::value
			&& ! std::is_same< OutputType, void >::value >::type
		> {
			static_assert(
				!( descr & descriptors::use_index ),
				"use_index descriptor given while output type is not numeric"
			);
			static_assert(
				std::is_convertible< D, OutputType >::value,
				"Cannot convert input to the given output type"
			);

		public:

			static OutputType getFromScalar( const D &x, const size_t ) noexcept {
				return static_cast< OutputType >( x );
			}
		};

	} // namespace internal

} // namespace alp

#endif // end ``_H_ALP_BLAS0''
