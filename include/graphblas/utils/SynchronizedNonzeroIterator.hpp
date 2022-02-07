
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
 * @date 2nd of August, 2017
 */

#ifndef _H_SYNCHRONIZEDNONZEROITERATOR
#define _H_SYNCHRONIZEDNONZEROITERATOR

#include <utility> //std::pair

#include <assert.h>

#ifdef _DEBUG
#ifndef _GRB_NO_STDIO
#include <iostream>
#endif
#endif

namespace grb {
	namespace utils {

	  template<typename T1, typename T2, typename T3> class common_it_tag { public: using it_tag = std::forward_iterator_tag ; };
	  template<> class common_it_tag<std::random_access_iterator_tag,
					 std::random_access_iterator_tag,
					 std::random_access_iterator_tag> { public: using it_tag = std::random_access_iterator_tag; };
	  

	  
	  template< typename S1,
				  typename S2,
				  typename V,
				  typename fwd_it1,
				  typename fwd_it2,
				  typename fwd_it3,
				  class it_type_calss = std::forward_iterator_tag>
		class SynchronizedNonzeroIterator {

			template< typename X1,
				  typename X2,
				  typename X3,
				  typename X4,
				  typename X5,
				  typename X6 >
			friend std::ostream & operator<<( std::ostream &, const SynchronizedNonzeroIterator< X1, X2, X3, X4, X5, X6 > & );

		private:
			// iterators to synchronise:
			fwd_it1 row_it, row_end;
			fwd_it2 col_it, col_end;
			fwd_it3 val_it, val_end;

			/** The currently active nonzero. */
			mutable std::pair< std::pair< S1, S2 >, V > nonzero;

			/** Whether #nonzero is up to date. */
			mutable bool updated;

			/** Updates the #nonzero fields using the current iterator values. */
			inline void update() const {
				assert( row_it != row_end );
				assert( col_it != col_end );
				assert( val_it != val_end );
				nonzero.first.first = *row_it;
				nonzero.first.second = *col_it;
				nonzero.second = *val_it;
				updated = true;
			}

		public:
			// STL typedefs:
			typedef std::ptrdiff_t difference_type;
			typedef std::pair< std::pair< S1, S2 >, V > value_type;
			typedef value_type & reference;
			typedef value_type * pointer;
		        typedef std::forward_iterator_tag iterator_category;


			// GraphBLAS typedefs:
			typedef S1 row_coordinate_type;
			typedef S2 column_coordinate_type;
			typedef V nonzero_value_type;

			/** Base constructor. Takes three sub-iterators as arguments. */
			SynchronizedNonzeroIterator( fwd_it1 it1,
						     fwd_it2 it2,
						     fwd_it3 it3,
						     fwd_it1 it1_end,
						     fwd_it2 it2_end,
						     fwd_it3 it3_end ) :
				row_it( it1 ), row_end( it1_end ), col_it( it2 ), col_end( it2_end ), val_it( it3 ), val_end( it3_end ), updated( false ) {
				if( it1 != it1_end && it2 != it2_end && it3 != it3_end ) {
					update();
					updated = false;
				}
			}

			/** Copy constructor. */
			SynchronizedNonzeroIterator( const SynchronizedNonzeroIterator< S1, S2, V, fwd_it1, fwd_it2, fwd_it3 > & other ) :
				row_it( other.row_it ), row_end( other.row_end ), col_it( other.col_it ), col_end( other.col_end ), val_it( other.val_it ), val_end( other.val_end ), updated( other.updated ) {
				if( updated && row_it != row_end && col_it != col_end && val_it != val_end ) {
					update();
				}
			}

			/** Assignment operator. */
			SynchronizedNonzeroIterator< S1, S2, V, fwd_it1, fwd_it2, fwd_it3 > & operator=( const SynchronizedNonzeroIterator< S1, S2, V, fwd_it1, fwd_it2, fwd_it3 > & other ) {
				row_it = other.row_it;
				row_end = other.row_end;
				col_it = other.col_it;
				col_end = other.col_end;
				val_it = other.val_it;
				val_end = other.val_end;
				updated = other.updated;
				if( updated && row_it != row_end && col_it != col_end && val_it != val_end ) {
					update();
				}
				return *this;
			}

			/** Equality check. */
			bool operator==( const SynchronizedNonzeroIterator< S1, S2, V, fwd_it1, fwd_it2, fwd_it3 > & other ) const {
				return row_it == other.row_it && col_it == other.col_it && val_it == other.val_it;
			}

			/** Inequality check. */
			bool operator!=( const SynchronizedNonzeroIterator< S1, S2, V, fwd_it1, fwd_it2, fwd_it3 > & other ) const {
				return row_it != other.row_it || col_it != other.col_it || val_it != other.val_it;
			};

			/** Increment operator. */
			SynchronizedNonzeroIterator< S1, S2, V, fwd_it1, fwd_it2, fwd_it3 > & operator++() {
				(void)++row_it;
				(void)++col_it;
				(void)++val_it;
				updated = false;
				return *this;
			}


			/** Direct derefence operator. */
			reference operator*() const {
				if( ! updated ) {
					assert( row_it != row_end && col_it != col_end && val_it != val_end );
					update();
				}
				return nonzero;
			}

			/** Pointer update. */
			const std::pair< std::pair< S1, S2 >, V > * operator->() const {
				if( ! updated ) {
					assert( row_it != row_end && col_it != col_end && val_it != val_end );
					update();
				}
				return &nonzero;
			}

			/** Returns the row coordinate. */
			const S1 & i() const {
				if( ! updated ) {
					assert( row_it != row_end && col_it != col_end && val_it != val_end );
					update();
				}
				return nonzero.first.first;
			}

			/** Returns the column coordinate. */
			const S2 & j() const {
				if( ! updated ) {
					assert( row_it != row_end && col_it != col_end && val_it != val_end );
					update();
				}
				return nonzero.first.second;
			}

			/** Returns the nonzero coordinate. */
			const V & v() const {
				if( ! updated ) {
					assert( row_it != row_end && col_it != col_end && val_it != val_end );
					update();
				}
				return nonzero.second;
			}
		};

			template< typename S1,
				  typename S2,
				  typename V,
				  typename fwd_it1,
				  typename fwd_it2,
				  typename fwd_it3>
	  class SynchronizedNonzeroIterator< S1,
					     S2,
					     V,
					     fwd_it1,
					     fwd_it2,
					     fwd_it3,
					     std::random_access_iterator_tag >
	  {

			template< typename X1,
				  typename X2,
				  typename X3,
				  typename X4,
				  typename X5,
				  typename X6,
				  typename X7>
			friend std::ostream & operator<<( std::ostream &, const SynchronizedNonzeroIterator< X1, X2, X3, X4, X5, X6, X7 > & );

		private:
			// iterators to synchronise:
			fwd_it1 row_it, row_end;
			fwd_it2 col_it, col_end;
			fwd_it3 val_it, val_end;

			/** The currently active nonzero. */
			mutable std::pair< std::pair< S1, S2 >, V > nonzero;

			/** Whether #nonzero is up to date. */
			mutable bool updated;

			/** Updates the #nonzero fields using the current iterator values. */
			inline void update() const {
				assert( row_it != row_end );
				assert( col_it != col_end );
				assert( val_it != val_end );
				nonzero.first.first = *row_it;
				nonzero.first.second = *col_it;
				nonzero.second = *val_it;
				updated = true;
			}

		public:
			// STL typedefs:
			typedef std::ptrdiff_t difference_type;
			typedef std::pair< std::pair< S1, S2 >, V > value_type;
			typedef value_type & reference;
			typedef value_type * pointer;
		        typedef std::random_access_iterator_tag iterator_category;


			// GraphBLAS typedefs:
			typedef S1 row_coordinate_type;
			typedef S2 column_coordinate_type;
			typedef V nonzero_value_type;

			/** Base constructor. Takes three sub-iterators as arguments. */
			SynchronizedNonzeroIterator( fwd_it1 it1,
						     fwd_it2 it2,
						     fwd_it3 it3,
						     fwd_it1 it1_end,
						     fwd_it2 it2_end,
						     fwd_it3 it3_end ) :
				row_it( it1 ), row_end( it1_end ), col_it( it2 ), col_end( it2_end ), val_it( it3 ), val_end( it3_end ), updated( false ) {
				if( it1 != it1_end && it2 != it2_end && it3 != it3_end ) {
					update();
					updated = false;
				}
			}

			/** Copy constructor. */
	    SynchronizedNonzeroIterator( const SynchronizedNonzeroIterator< S1, S2, V, fwd_it1, fwd_it2, fwd_it3,std::random_access_iterator_tag > & other ) :
				row_it( other.row_it ), row_end( other.row_end ), col_it( other.col_it ), col_end( other.col_end ), val_it( other.val_it ), val_end( other.val_end ), updated( other.updated ) {
				if( updated && row_it != row_end && col_it != col_end && val_it != val_end ) {
					update();
				}
			}

			/** Assignment operator. */
			SynchronizedNonzeroIterator
			< S1, S2, V, fwd_it1, fwd_it2, fwd_it3,std::random_access_iterator_tag > &
			operator=( const SynchronizedNonzeroIterator< S1, S2, V, fwd_it1, fwd_it2, fwd_it3,std::random_access_iterator_tag > & other ) {
				row_it = other.row_it;
				row_end = other.row_end;
				col_it = other.col_it;
				col_end = other.col_end;
				val_it = other.val_it;
				val_end = other.val_end;
				updated = other.updated;
				if( updated && row_it != row_end && col_it != col_end && val_it != val_end ) {
					update();
				}
				return *this;
			}

			/** Assignment operator. */
			/** Increment operator. */
			SynchronizedNonzeroIterator
			< S1, S2, V, fwd_it1, fwd_it2, fwd_it3, std::random_access_iterator_tag > & operator+=( const std::size_t inc ) {
				row_it+=inc;
				col_it+=inc;
				val_it+=inc;
				updated = false;
				return *this;
			}
	    
			/** Equality check. */
			bool operator==( const SynchronizedNonzeroIterator
					 < S1, S2, V, fwd_it1, fwd_it2, fwd_it3,std::random_access_iterator_tag > & other ) const {
				return row_it == other.row_it && col_it == other.col_it && val_it == other.val_it;
			}

			/** difference operator. */
			std::size_t  operator-( const SynchronizedNonzeroIterator
				       < S1, S2, V, fwd_it1, fwd_it2, fwd_it3,std::random_access_iterator_tag > & other ) const {
			  return row_it - other.row_it;
			}	    

			/** Inequality check. */
			bool operator!=( const SynchronizedNonzeroIterator
					 < S1, S2, V, fwd_it1, fwd_it2, fwd_it3, std::random_access_iterator_tag > & other ) const {
				return row_it != other.row_it || col_it != other.col_it || val_it != other.val_it;
			};

			/** Increment operator. */
			SynchronizedNonzeroIterator
			< S1, S2, V, fwd_it1, fwd_it2, fwd_it3, std::random_access_iterator_tag > & operator++() {
				(void)++row_it;
				(void)++col_it;
				(void)++val_it;
				updated = false;
				return *this;
			}

			/** Direct derefence operator. */
			reference operator*() const {
				if( ! updated ) {
					assert( row_it != row_end && col_it != col_end && val_it != val_end );
					update();
				}
				return nonzero;
			}

			/** Pointer update. */
			const std::pair< std::pair< S1, S2 >, V > * operator->() const {
				if( ! updated ) {
					assert( row_it != row_end && col_it != col_end && val_it != val_end );
					update();
				}
				return &nonzero;
			}

			/** Returns the row coordinate. */
			const S1 & i() const {
				if( ! updated ) {
					assert( row_it != row_end && col_it != col_end && val_it != val_end );
					update();
				}
				return nonzero.first.first;
			}

			/** Returns the column coordinate. */
			const S2 & j() const {
				if( ! updated ) {
					assert( row_it != row_end && col_it != col_end && val_it != val_end );
					update();
				}
				return nonzero.first.second;
			}

			/** Returns the nonzero coordinate. */
			const V & v() const {
				if( ! updated ) {
					assert( row_it != row_end && col_it != col_end && val_it != val_end );
					update();
				}
				return nonzero.second;
			}
		};


	  

			  

		template< typename S1,
			  typename S2,
			  typename fwd_it1,
			  typename fwd_it2 >
		class SynchronizedNonzeroIterator< S1,
						   S2, void,
						   fwd_it1,
						   fwd_it2,
						   void
						   > {
			template< typename X1, typename X2, typename X3, typename X4 >
			friend std::ostream & operator<<( std::ostream &, const SynchronizedNonzeroIterator< X1, X2, void, X3, X4, void > & );

		private:
			// iterators to synchronise:
			fwd_it1 row_it, row_end;
			fwd_it2 col_it, col_end;

			typedef typename std::iterator_traits<fwd_it1>::iterator_category iterator1_category ;
			typedef typename std::iterator_traits<fwd_it2>::iterator_category iterator2_category ;
		  

			/** The currently active nonzero. */
			mutable std::pair< S1, S2 > nonzero;

			/** Whether #nonzero is up to date. */
			mutable bool updated;

			/** Updates the #nonzero fields using the current iterator values. */
			inline void update() const {
				assert( row_it != row_end );
				assert( col_it != col_end );
				nonzero.first = *row_it;
				nonzero.second = *col_it;
				updated = true;
			}

		public:
			// STL typedefs:
			typedef std::ptrdiff_t difference_type;
			typedef std::pair< S1, S2 > value_type;
			typedef value_type & reference;
			typedef value_type * pointer;
		        typedef std::forward_iterator_tag iterator_category;

			// GraphBLAS typedefs:
			typedef S1 row_coordinate_type;
			typedef S2 column_coordinate_type;
			typedef void nonzero_value_type;

			/** Base constructor. Takes two sub-iterators as arguments. */
			SynchronizedNonzeroIterator( fwd_it1 it1, fwd_it2 it2, fwd_it1 it1_end, fwd_it2 it2_end ) :
			  row_it( it1 ), row_end( it1_end ), col_it( it2 ), col_end( it2_end ), updated( false ) {
				if( it1 != it1_end && it2 != it2_end ) {
					update();
				}
			}

			/** Copy constructor. */
			SynchronizedNonzeroIterator( const SynchronizedNonzeroIterator< S1, S2, void, fwd_it1, fwd_it2, void > & other ) :
			  row_it( other.row_it ), row_end( other.row_end ), col_it( other.col_it ), col_end( other.col_end ), updated( other.updated ) {
				if( updated && row_it != row_end && col_it != col_end ) {
					update();
				}
			}

			/** Assignment operator. */
			SynchronizedNonzeroIterator< S1, S2, void, fwd_it1, fwd_it2, void > &
			operator=( const SynchronizedNonzeroIterator< S1, S2, void, fwd_it1, fwd_it2, void > & other ) {
				row_it = other.row_it;
				row_end = other.row_end;
				col_it = other.col_it;
				col_end = other.col_end;
				updated = other.updated;
				if( updated && row_it != row_end && col_it != col_end ) {
					update();
				}
				return *this;
			}

			/** Equality check. */
			bool operator==( const SynchronizedNonzeroIterator< S1, S2, void, fwd_it1, fwd_it2, void > & other ) const {
				return row_it == other.row_it && col_it == other.col_it;
			}

			/** Inequality check. */
			bool operator!=( const SynchronizedNonzeroIterator< S1, S2, void, fwd_it1, fwd_it2, void > & other ) const {
				return row_it != other.row_it || col_it != other.col_it;
			};

			/** Increment operator. */
			SynchronizedNonzeroIterator< S1, S2, void, fwd_it1, fwd_it2, void > & operator++() {
				(void)++row_it;
				(void)++col_it;
				updated = false;
				return *this;
			}

			/** Direct derefence operator. */
			reference operator*() const {
				if( ! updated ) {
					assert( row_it != row_end && col_it != col_end );
					update();
				}
				return nonzero;
			}

			/** Pointer update. */
			const std::pair< S1, S2 > * operator->() const {
				if( ! updated ) {
					assert( row_it != row_end && col_it != col_end );
					update();
				}
				return &nonzero;
			}

			/** Returns the row coordinate. */
			const S1 & i() const {
				if( ! updated ) {
					assert( row_it != row_end && col_it != col_end );
					update();
				}
				return nonzero.first;
			}

			/** Returns the column coordinate. */
			const S2 & j() const {
				if( ! updated ) {
					assert( row_it != row_end && col_it != col_end );
					update();
				}
				return nonzero.second;
			}
		};


	  

		template< typename S1, typename S2, typename V, typename fwd_it1, typename fwd_it2, typename fwd_it3 >
		std::ostream & operator<<( std::ostream & os, const SynchronizedNonzeroIterator< S1, S2, V, fwd_it1, fwd_it2, fwd_it3 > & it ) {
			if( ! it.updated ) {
				it.update();
			}
			os << it.nonzero.first.first << ", " << it.nonzero.first.second << ", " << it.nonzero.second;
			return os;
		}

		template< typename S1, typename S2, typename fwd_it1, typename fwd_it2 >
		std::ostream & operator<<( std::ostream & os, const SynchronizedNonzeroIterator< S1, S2, void, fwd_it1, fwd_it2, void > & it ) {
			if( ! it.updated ) {
				it.update();
			}
			os << it.nonzero.first << ", " << it.nonzero.second;
			return os;
		}

		template< typename S1,
			  typename S2 >
		SynchronizedNonzeroIterator< S1,
					     S2, void,
					     const S1 *,
					     const S2 *, void >
		makeSynchronized( const S1 * const it1,
				  const S2 * const it2,
				  const S1 * const it1_end,
				  const S2 * const it2_end ) {
#ifdef _DEBUG
#ifndef _GRB_NO_STDIO
			std::cout << "SynchronizedNonzeroIterator::makeSynchronized "
						 "received iterators "
					  << it1 << " (start) and " << it2 << " (end)\n";
#else
			printf( "SynchronizedNonzeroIterator::makeSynchronized received "
					"iterators %p (start) and %p (end)\n",
				it1, it2 );
#endif
#endif
			return SynchronizedNonzeroIterator< S1, S2, void, const S1 *, const S2 *, void >( it1, it2, it1_end, it2_end );
		}

		template< typename fwd_it1, typename fwd_it2 >
		SynchronizedNonzeroIterator< typename fwd_it1::value_type,
					     typename fwd_it2::value_type,
					     void,
					     fwd_it1,
					     fwd_it2,
					     void >
		makeSynchronized( const fwd_it1 it1,
				  const fwd_it2 it2,
				  const fwd_it1 it1_end,
				  const fwd_it2 it2_end ) {
			return SynchronizedNonzeroIterator< typename fwd_it1::value_type, typename fwd_it2::value_type, void, fwd_it1, fwd_it2, void >( it1, it2, it1_end, it2_end );
		}


		template< typename S1,
			  typename S2,
			  typename V,
			  typename ITtag>
		SynchronizedNonzeroIterator<S1,
					    S2,
					    V,
					    const S1 *,
					    const S2 *,
					    const V *,
					    ITtag >
		makeSynchronized( const S1 * const it1,
				  const S2 * const it2,
				  const V  * const it3,
				  const S1 * const it1_end,
				  const S2 * const it2_end,
				  const V  * const it3_end,
				  ITtag  ) {

		  return SynchronizedNonzeroIterator
		    < S1, S2, V, const S1 *, const S2 *, const V *, ITtag>
		    (it1,it2,it3,it1_end,it2_end,it3_end);
		}

	  
		template< typename fwd_it1,
			  typename fwd_it2,
			  typename fwd_it3 >
		SynchronizedNonzeroIterator< typename fwd_it1::value_type,
					     typename fwd_it2::value_type,
					     typename fwd_it3::value_type,
					     fwd_it1,
					     fwd_it2,
					     fwd_it3 >
		makeSynchronized( const fwd_it1 it1,
				  const fwd_it2 it2,
				  const fwd_it3 it3,
				  const fwd_it1 it1_end,
				  const fwd_it2 it2_end,
				  const fwd_it3 it3_end ) {
		  return SynchronizedNonzeroIterator
		    < typename fwd_it1::value_type, typename fwd_it2::value_type, typename fwd_it3::value_type, fwd_it1, fwd_it2, fwd_it3 >
		    (it1, it2, it3, it1_end, it2_end, it3_end );
		}

	} // namespace utils
} // namespace grb

#endif // end ``_H_SYNCHRONIZEDNONZEROITERATOR''
