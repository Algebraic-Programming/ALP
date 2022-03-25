
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
 * @date 11th of August, 2016
 */

#ifndef _H_ALP_IDENTITIES
#define _H_ALP_IDENTITIES

#include <limits>

namespace alp {

	/**
	 * Standard identities common to many operators.
	 *
	 * The most commonly used identities are
	 *   - #alp::identities::zero, and
	 *   - #alp::identities::one.
	 *
	 * A stateful identity should expose the same public interface as the
	 * identities collected here, which is class which exposes at least one public
	 * templated function named \a value, taking no arguments, returning the
	 * identity in the domain \a D. This type \a D is the first template parameter
	 * of the function \a value. If there are other template parameters, those
	 * template parameters are required to have defaults.
	 *
	 * @see operators
	 * @see Monoid
	 * @see Semiring
	 */
	namespace identities {

		/** Standard identity for numerical addition. */
		template< typename D >
		class zero {
			static_assert( std::is_convertible< int, D >::value, "Cannot form identity under the requested domain" );

		public:
			/**
			 * @tparam D The domain of the value to return.
			 * @return The identity under standard addition (i.e., `zero').
			 */
			static constexpr D value() {
				return static_cast< D >( 0 );
			}
		};
		template< typename K, typename V >
		class zero< std::pair< K, V > > {
		public:
			static constexpr std::pair< K, V > value() {
				return std::make_pair( zero< K >::value(), zero< V >::value() );
			}
		};

		/** Standard identity for numerical multiplication. */
		template< typename D >
		class one {
			static_assert( std::is_convertible< int, D >::value, "Cannot form identity under the requested domain" );

		public:
			/**
			 * @tparam D The domain of the value to return.
			 * @return The identity under standard multiplication (i.e., `one').
			 */
			static constexpr D value() {
				return static_cast< D >( 1 );
			}
		};
		template< typename K, typename V >
		class one< std::pair< K, V > > {
		public:
			static constexpr std::pair< K, V > value() {
				return std::make_pair( one< K >::value(), one< V >::value() );
			}
		};

		/** Standard identity for the minimum operator. */
		template< typename D >
		class infinity {
			static_assert( std::is_arithmetic< D >::value, "Cannot form identity under the requested domain" );

		public:
			/**
			 * @tparam D The domain of the value to return.
			 * @return The identity under the standard min operator (i.e., `infinity'),
			 *         of type \a D.
			 */
			static constexpr D value() {
				return std::numeric_limits< D >::has_infinity ? std::numeric_limits< D >::infinity() : std::numeric_limits< D >::max();
			}
		};
		template< typename K, typename V >
		class infinity< std::pair< K, V > > {
		public:
			static constexpr std::pair< K, V > value() {
				return std::make_pair( infinity< K >::value(), infinity< V >::value() );
			}
		};

		/** Standard identity for the maximum operator. */
		template< typename D >
		class negative_infinity {
			static_assert( std::is_arithmetic< D >::value, "Cannot form identity under the requested domain" );

		public:
			/**
			 * @tparam D The domain of the value to return.
			 * @return The identity under the standard max operator, i.e.,
			 *         `minus infinity'.
			 */
			static constexpr D value() {
				return std::numeric_limits< D >::min() == 0 ? 0 : ( std::numeric_limits< D >::has_infinity ? -std::numeric_limits< D >::infinity() : std::numeric_limits< D >::min() );
			}
		};
		template< typename K, typename V >
		class negative_infinity< std::pair< K, V > > {
		public:
			static constexpr std::pair< K, V > value() {
				return std::make_pair( negative_infinity< K >::value(), negative_infinity< V >::value() );
			}
		};

		/**
		 * Standard identitity for the logical or operator.
		 *
		 * @see operators::logical_or.
		 */
		template< typename D >
		class logical_false {
			static_assert( std::is_convertible< bool, D >::value, "Cannot form identity under the requested domain" );

		public:
			/**
			 * @tparam D The domain of the value to return.
			 * @return The identity under the standard logical OR operator, i.e.,
			 *         \a false.
			 */
			static const constexpr D value() {
				return static_cast< D >( false );
			}
		};
		template< typename K, typename V >
		class logical_false< std::pair< K, V > > {
		public:
			static constexpr std::pair< K, V > value() {
				return std::make_pair( logical_false< K >::value(), logical_false< V >::value() );
			}
		};

		/**
		 * Standard identity for the logical AND operator.
		 *
		 * @see operators::logical_and.
		 */
		template< typename D >
		class logical_true {
			static_assert( std::is_convertible< bool, D >::value, "Cannot form identity under the requested domain" );

		public:
			/**
			 * @tparam D The domain of the value to return.
			 * @return The identity under the standard logical AND operator, i.e.,
			 *         \a true.
			 */
			static constexpr D value() {
				return static_cast< D >( true );
			}
		};
		template< typename K, typename V >
		class logical_true< std::pair< K, V > > {
		public:
			static constexpr std::pair< K, V > value() {
				return std::make_pair( logical_true< K >::value(), logical_true< V >::value() );
			}
		};

	} // namespace identities
} // namespace alp

#endif
