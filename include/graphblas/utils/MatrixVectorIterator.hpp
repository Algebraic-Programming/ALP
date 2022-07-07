
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
 * @author V. Vlacic
 */

#ifndef _H_MATRIXVECTORITERATOR
#define _H_MATRIXVECTORITERATOR

#include <cstddef> //std::ptrdiff_t
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept> //std::runtime_error

#include <stdlib.h> //posix_memalign

#include <utility> //std::pair, std::move, std::forward

#include <assert.h>

#include <graphblas/config.hpp>
#include <graphblas/iomode.hpp>
#include <graphblas/utils/config.hpp>
#include <graphblas/vector.hpp>


namespace grb {

	namespace utils {

		namespace internal {

			/** TODO documentation */
			template<
				typename T, typename V,
				typename SR = grb::config::RowIndexType,
				typename SC = grb::config::ColIndexType,
				Backend backend = grb::config::default_backend
			>
			class MatrixVectorIteratorBase {

				protected:

					/** The generic output type */
					typedef std::pair< std::pair< SR, SC >, T > OutputType;

					/** The type of the input iterator */
					typedef typename Vector< V, backend >::const_iterator InputIteratorType;

					/** A default identity mapper function. */
					static OutputType identity( const size_t &index, const V &value ) {
						const auto coors = std::make_pair( index, index );
						return std::make_pair( coors, value );
					}

					/** The input Vector iterator */
					InputIteratorType vector_iterator;

					/** The current output */
					mutable OutputType current;

					/** Whether current is initialised */
					mutable bool initialized;

					/** A function to apply to convert the vector entries to the matrix entries. */
					const std::function< OutputType( const size_t &, const V & ) > converter;

					/** Main constructor */
					MatrixVectorIteratorBase(
						InputIteratorType vec_iter,
						const std::function< OutputType( const size_t &, const V & ) > conv
					) noexcept :
						vector_iterator( std::move( vec_iter ) ),
						initialized( false ),
						converter( conv )
					{}

					/** Copy constructor */
					MatrixVectorIteratorBase(
						const MatrixVectorIteratorBase &other
					) noexcept :
						vector_iterator( other.vector_iterator ),
						current( other.current ),
						initialized( other.initialized ),
						converter( other.converter )
					{}

					/** Move constructor */
					MatrixVectorIteratorBase( MatrixVectorIteratorBase &&other ) noexcept :
						vector_iterator( std::move( other.vector_iterator ) ),
						current( std::move( other.current ) ),
						initialized( std::move( other.initialized ) ),
						converter( std::move( other.converter ) )
					{}

					/** Default constructor */
					MatrixVectorIteratorBase() noexcept :
						vector_iterator( Vector< V, backend >().cbegin() ),
						initialized( false ),
						converter( identity )
					{}

					/** Reads out vector_iterator and stores result in output */
					void setValue() const { // no noexcept because converter could throw
						assert( converter != nullptr );
						current = converter(
							vector_iterator->first,
							vector_iterator->second
						);
					}


				public:

					/** Returns the current nonzero value. */
					const T v() const {
						if( !initialized ) {
							setValue();
							initialized = true;
						}
						return current.second;
					}

					/** Returns the current row index. */
					const SR i() const {
						if( !initialized ) {
							setValue();
							initialized = true;
						}
						return current.first.first;
					}

					/** Returns the current column index. */
					const SC j() const {
						if( !initialized ) {
							setValue();
							initialized = true;
						}
						return current.first.second;
					}

			};

			/** Pattern matrix specialisation */
			template< typename V, typename SR, typename SC, Backend backend >
			class MatrixVectorIteratorBase< void, V, SR, SC, backend > {

				protected:

					/** The generic output type */
					typedef std::pair< SR, SC > OutputType;

					/** The type of the input iterator */
					typedef typename Vector< V, backend >::const_iterator InputIteratorType;

					/** A default identity mapper function. */
					static OutputType identity( const size_t &index, const V & ) {
						return std::make_pair( index, index );
					}

					/** The input Vector iterator */
					InputIteratorType vector_iterator;

					/** The current output */
					mutable OutputType current;

					/** Whether current is initialised */
					mutable bool initialized;

					/** A function to apply to convert the vector entries to the matrix entries. */
					const std::function< OutputType( const size_t &, const V & ) > converter;

					/** Main constructor */
					MatrixVectorIteratorBase(
						const typename Vector< V, backend >::const_iterator vec_iter,
						const std::function< OutputType( const size_t &, const V & ) > conv
					) noexcept :
						vector_iterator( std::move( vec_iter ) ),
						initialized( false ),
						converter( std::move( conv ) )
					{}

					/** Copy constructor */
					MatrixVectorIteratorBase(
						const MatrixVectorIteratorBase &other
					) noexcept :
						vector_iterator( other.vector_iterator ),
						current( other.current ),
						initialized( other.initialized ),
						converter( other.converter )
					{}

					/** Move constructor */
					MatrixVectorIteratorBase( MatrixVectorIteratorBase &&other ) noexcept :
						vector_iterator( std::move( other.vector_iterator ) ),
						current( std::move( other.current ) ),
						initialized( std::move( other.initialized ) ),
						converter( std::move( other.converter ) )
					{}

					/** Default constructor */
					MatrixVectorIteratorBase() noexcept :
						vector_iterator( InputIteratorType() ),
						initialized( false ),
						converter( identity )
					{}

					/** Reads out vector_iterator and stores result in output */
					void setValue() const { // no noexcept since converter could throw
						assert( converter != nullptr );
						current = converter( vector_iterator->first, vector_iterator->second );
					}


			public:

					/** Returns the current row index. */
					const SR i() const {
						if( !initialized ) {
							setValue();
							initialized = true;
						}
						return current.first;
					}

					/** Returns the current column index. */
					const SC j() const {
						if( !initialized ) {
							setValue();
							initialized = true;
						}
						return current.second;
					}

			};

		} // namespace internal

		/** TODO documentation */
		template<
			typename T, typename V,
			typename SR = grb::config::RowIndexType,
			typename SC = grb::config::ColIndexType,
			Backend backend = grb::config::default_backend
		>
		class MatrixVectorIterator :
			public internal::MatrixVectorIteratorBase< T, V, SR, SC, backend >
		{
			using typename internal::MatrixVectorIteratorBase< T, V, SR, SC, backend >::OutputType;
			using internal::MatrixVectorIteratorBase< T, V, SR, SC, backend >::vector_iterator;
			using internal::MatrixVectorIteratorBase< T, V, SR, SC, backend >::current;
			using internal::MatrixVectorIteratorBase< T, V, SR, SC, backend >::initialized;
			using internal::MatrixVectorIteratorBase< T, V, SR, SC, backend >::converter;
			using internal::MatrixVectorIteratorBase< T, V, SR, SC, backend >::setValue;

		public:

			/** The return type of i(). */
			typedef SR RowIndexType;

			/** The return type of j(). */
			typedef SC ColumnIndexType;

			/** The return type of v() or <tt>void</tt>. */
			typedef T ValueType;

			/** The STL iterator output type. */
			typedef OutputType value_type;

			/** main constructor */
			MatrixVectorIterator(
				const typename Vector< V, backend >::const_iterator vec_iter,
				const std::function< OutputType( const size_t &, const V & ) > conv
			) noexcept :
				internal::MatrixVectorIteratorBase< T, V, SR, SC, backend >(
					std::move( vec_iter ), std::move( conv )
				)
			{}

			/** default constructor */
			MatrixVectorIterator() noexcept :
				internal::MatrixVectorIteratorBase< T, V, SR, SC, backend >()
			{}

			/** copy constructor */
			MatrixVectorIterator( const MatrixVectorIterator &other ) noexcept :
				internal::MatrixVectorIteratorBase< T, V, SR, SC, backend >( other )
			{}

			/** move constructor */
			MatrixVectorIterator( MatrixVectorIterator &&other ) noexcept :
				internal::MatrixVectorIteratorBase< T, V, SR, SC, backend >(
					std::move( other )
				)
			{}

			/** copy assignment */
			MatrixVectorIterator & operator=( const MatrixVectorIterator &x ) noexcept {
				vector_iterator = x.vector_iterator;
				converter = x.converter;
				return *this;
			}

			/** move assignment */
			MatrixVectorIterator & operator=( MatrixVectorIterator &&x ) noexcept {
				vector_iterator = std::move( x.vector_iterator );
				converter = std::move( x.converter );
				return *this;
			}

			/** check for equality */
			bool operator==( const MatrixVectorIterator &x ) const noexcept {
				return vector_iterator == x.vector_iterator;
			}

			/** standard check for inequality, relies on equality check */
			bool operator!=( const MatrixVectorIterator &x ) const noexcept {
				return !( operator==( x ) );
			}

			/** standard increment iterator */
			MatrixVectorIterator & operator++() {
				(void)++vector_iterator;
				setValue();
				return *this;
			}

			/** standard dereferencing of iterator */
			const OutputType & operator*() const {
				if( !initialized ) {
					setValue();
					initialized = true;
				}
				return current;
			}

			/** standard pointer request of iterator */
			const OutputType * operator->() const {
				if( !initialized ) {
					setValue();
					initialized = true;
				}
				return &current;
			}

		};

		namespace internal {

			// vector to matrix converter base
			template<
				typename T, typename V,
				Backend backend = grb::config::default_backend
			>
			class VectorToMatrixConverterBase {

				protected:

					/** Alias of the underlying iterator. */
					typedef typename Vector< V, backend >::const_iterator SourceIteratorType;

					/** Get local alias to the row index type. */
					typedef typename grb::config::RowIndexType SR;

					/** Get local alias to the column index type. */
					typedef typename grb::config::ColIndexType SC;

					/** Element type of the output iterator. */
					typedef std::pair< std::pair< SR, SC >, T > OutputType;

					/** Source iterator in start position. */
					const SourceIteratorType src_start;

					/** Source iterator in end position. */
					const SourceIteratorType src_end;

					/** The converter function. */
					const std::function< OutputType( const size_t &, const V & ) > converter;

					/** Main constructor */
					VectorToMatrixConverterBase(
						const SourceIteratorType start, const SourceIteratorType end,
						const std::function< OutputType( const size_t &, const V & ) > conv
					) noexcept :
						src_start( std::move( start ) ),
						src_end( std::move( end ) ),
						converter( std::move( conv ) )
					{
#ifdef _DEBUG
						std::cout << "\t in VectorToMatrixConverterBase constructor for non-void matrices\n";
#endif
					}

			};

			// vector to matrix converter base -- pattern matrix specialisation
			template< typename V, Backend backend >
			class VectorToMatrixConverterBase< void, V, backend > {


				protected:

					/** Alias of the underlying iterator. */
					typedef typename Vector< V, backend >::const_iterator SourceIteratorType;

					/** Get local alias to the row index type. */
					typedef typename grb::config::RowIndexType SR;

					/** Get local alias to the column index type. */
					typedef typename grb::config::ColIndexType SC;

					/** Value type of the output iterator */
					typedef std::pair< SR, SC > OutputType;

					/** The source iterator in start position */
					const SourceIteratorType src_start;

					/** The source iterator in end position. */
					const SourceIteratorType src_end;

					/** The converter function. */
					const std::function< OutputType( const size_t &, const V & ) > converter;

					/** Main constructor. */
					VectorToMatrixConverterBase(
						const SourceIteratorType start, const SourceIteratorType end,
						const std::function< OutputType( const size_t &, const V & ) > conv
					) noexcept :
						src_start( std::move( start ) ),
						src_end( std::move( end ) ),
						converter( std::move( conv ) )
					{
#ifdef _DEBUG
						std::cout << "In VectorToMatrixConverterBase constructor "
							<< "for void matrices\n";
#endif
					}

			};

		} // namespace internal

		/** TODO documentation */
		template<
			typename T, typename V,
			Backend backend = grb::config::default_backend
		>
		class VectorToMatrixConverter :
			internal::VectorToMatrixConverterBase< T, V, backend >
		{
			using typename internal::VectorToMatrixConverterBase< T, V, backend >::OutputType;
			using internal::VectorToMatrixConverterBase< T, V, backend >::src_start;
			using internal::VectorToMatrixConverterBase< T, V, backend >::src_end;
			using internal::VectorToMatrixConverterBase< T, V, backend >::converter;

			private:

				/** Short-cut type for an iterator over vectors. */
				typedef typename grb::Vector< V, backend >::const_iterator VectorIterator;

				/** Get local typedef of the row index type. */
				typedef typename internal::VectorToMatrixConverterBase< T, V, backend >::SR SR;

				/** Get local typedef of the columnindex type. */
				typedef typename internal::VectorToMatrixConverterBase< T, V, backend >::SC SC;

			public:

				/** TODO documentation */
				VectorToMatrixConverter(
					const VectorIterator start, const VectorIterator end,
					const std::function< OutputType( const size_t &, const V & ) > conv
				) noexcept :
					internal::VectorToMatrixConverterBase< T, V, backend >(
						std::move( start ), std::move( end ), std::move( conv )
					)
				{
#ifdef _DEBUG
					std::cout << "In VectorToMatrixConverter constructor\n";
#endif
				}

				/**
				 * Does not throw exceptions if and only if the converter does not throw.
				 */
				MatrixVectorIterator< T, V, SR, SC, backend > cbegin() const {
					return MatrixVectorIterator< T, V, SR, SC, backend >(
						src_start, converter
					);
				}

				/**
				 * Does not throw exceptions if and only if the converter does not throw.
				 */
				MatrixVectorIterator< T, V, SR, SC, backend > cend() const {
					return MatrixVectorIterator< T, V, SR, SC, backend >(
						src_end, converter
					);
				}

				// we only provide const iterators

				/**
				 * Does not throw exceptions if and only if the converter does not throw.
				 */
				MatrixVectorIterator< T, V, SR, SC, backend > begin() const {
					return cbegin();
				}

				/**
				 * Does not throw exceptions if and only if the converter does not throw.
				 */
				MatrixVectorIterator< T, V, SR, SC, backend > end() const {
					return cend();
				}

		};

		/** TODO documentation */
		template<
			typename OutputType, typename InputType, Backend backend, class Converter
		>
		VectorToMatrixConverter< OutputType, InputType, backend >
		makeVectorToMatrixConverter(
			const grb::Vector< InputType, backend > &vec,
			const Converter converter
		) noexcept {
#ifdef _DEBUG
			std::cout << "In makeVectorToMatrixConverter\n";
#endif
			return VectorToMatrixConverter< OutputType, InputType, backend >(
				vec.cbegin(), vec.cend(), std::move( converter )
			);
		}

	} // namespace utils

} // namespace grb

#endif // end ``_H_MATRIXVECTORITERATOR''

