
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
 * @author Benjamin Lozes
 * @date 4th of September, 2023
 */

#ifndef _H_CHAINEDITERATORS
#define _H_CHAINEDITERATORS

#include <type_traits>
#include <iostream>
#include <vector>

namespace grb::utils::iterators {

    /**
     * A wrapper iterator around a sequence of iterators that allows
     * to iterate over all the iterators as if they were a single one.
     *
     * @tparam SubIterType The given of the sub-iterators
     * (must be an unique iterator type)
     */
    template< typename SubIterType >
    class ChainedIterators {

        private:
            // List of sub-iterators ranges [(begin, end), ...]
            std::vector< std::pair< SubIterType, SubIterType> > _iterators;

            // Index of the current sub-iterator
            size_t _current_iterator;

            // Index of the current element in the current sub-iterator
            size_t _current_subiter_index;

        public:
            // STL type definitions
            typedef typename std::iterator_traits< SubIterType >::value_type
                value_type;
            typedef typename std::iterator_traits< SubIterType >::pointer
                pointer;
            typedef typename std::iterator_traits< SubIterType >::reference
                reference;
            typedef typename std::iterator_traits< SubIterType >::iterator_category
                iterator_category;
            typedef typename std::iterator_traits< SubIterType >::difference_type
                difference_type;

            using self_type = ChainedIterators< SubIterType >;
            using const_self_type = const ChainedIterators< SubIterType >;

            /** The base constructor. */
            ChainedIterators() = delete;

            /** The base constructor. */
            ChainedIterators(
                const std::vector< std::pair< SubIterType, SubIterType > > &it,
                size_t current_iterator = 0UL,
                size_t current_subiter_index = 0UL
            ) :
                _iterators( it ),
                _current_iterator( current_iterator ),
                _current_subiter_index( current_subiter_index )
            {}

            /** The next operator. */
            self_type& operator++() {
                ++_current_subiter_index;

                // If reached the end of the current iterator, move to the next one
                if( _current_subiter_index >=
                    std::distance( _iterators[ _current_iterator ].first,
                        _iterators[ _current_iterator ].second )
                ) {
                    ++_current_iterator;
                    _current_subiter_index = 0;
                }

                return *this;
            }

            self_type operator++(int) noexcept {
                self_type tmp( *this );
                operator++();
                return tmp;
            }

            /** The minus operator */
            self_type& operator--() {
                    // If reached the beginning of the current iterator
                if( _current_subiter_index == 0 ) {

                    // If reached the beginning, do nothing
                    if( _current_iterator == 0 ) {
                        return *this;
                    }

                    // Move to the previous iterator
                    --_current_iterator; // Can not be zero here
                    _current_subiter_index = std::distance(
                        _iterators[ _current_iterator ].first,
                        _iterators[ _current_iterator ].second
                    ) - 1;
                } else {
                    --_current_subiter_index;
                }

                return *this;
            }

            self_type operator--(int) noexcept {
                self_type tmp( *this );
                operator--();
                return tmp;
            }

            /** The value operator. */
            value_type operator*() const {
                return *(
                    _iterators[ _current_iterator ].first
                        +  _current_subiter_index
                );
            }

            /** The arrow operator. */
            pointer operator->() const {
                return &( operator*() );
            }

            /** The equality operator. */
            bool operator==( const ChainedIterators &other ) const {
                return
                    _iterators == other._iterators &&
                    _current_iterator == other._current_iterator &&
                    _current_subiter_index == other._current_subiter_index;
            }

            /** The inequality operator. */
            bool operator!=( const ChainedIterators &other ) const {
                return !( *this == other );
            }

            friend self_type operator-(
                const_self_type & iterator,
                const size_t count
            ) noexcept {
                for( size_t i = 0; i < count; ++i ) {
                    --iterator;
                }
            }

            friend self_type operator-(
                const size_t count,
                const_self_type &iterator
            ) noexcept {
                return iterator - count;
            }

            difference_type operator-(
                const_self_type &iterator
            ) const noexcept {
                difference_type dist = std::distance(
                        iterator._iterators[iterator._current_iterator ].first,
                        iterator._iterators[iterator._current_iterator ].second
                    ) - iterator._current_subiter_index;
                const auto i_start = iterator._current_iterator + 1;
                const auto i_end = _current_iterator;
                for( size_t i = i_start; i < i_end; ++i ) {
                    dist += std::distance(
                        iterator._iterators[ i ].first,
                        iterator._iterators[ i ].second
                    );
                }
                dist += _current_subiter_index;
                return dist;
            }

            self_type& operator+=( const size_t count ) noexcept {
                for( size_t i = 0; i < count; ++i ) {
                    ++( *this );
                }
                return *this;
            }

            self_type& operator-=( const size_t count ) noexcept {
                for( size_t i = 0; i < count; ++i ) {
                    --( *this );
                }
                return *this;
            }

            difference_type do_distance(
                const_self_type &first, const_self_type &last, std::input_iterator_tag
            ) const noexcept {
                difference_type result = 0;
                while( first != last ) {
                    ++first;
                    ++result;
                }
                return result;
            }

            difference_type do_distance(
                const_self_type &first, const_self_type &last, std::random_access_iterator_tag
            ) const noexcept {
                return last - first;
            }
    };

} // namespace grb::utils::iterators

namespace grb::utils::containers {

    /**
     * A wrapper around a sequence of iterators that exposes a
     * single constant iterator of matching type.
     *
     * @tparam SubIterType The given of the sub-iterators (must be an iterator type)
     *
     * This declaration uses SFINAE in order to expose implementations for
     * supported value types only, based on the given \a SubIterType.
     */
    template< typename SubIterType >
    class ChainedIteratorsVector {

        private:

            /**
             * The vector of iterators to chain.
             * Each iterator is a pair of \a begin and \a end iterators'
             * values: [(begin, end), ...]
             */
            std::vector< std::pair< SubIterType, SubIterType> > _iterators;

        public:

            using iterator = utils::iterators::ChainedIterators< SubIterType >;
            using const_iterator = const utils::iterators::ChainedIterators< SubIterType >;

            /** The base constructor. */
            ChainedIteratorsVector( size_t capacity = 0UL ) {
                _iterators.reserve( capacity );
            }

            /** The base constructor. */
            ChainedIteratorsVector(
                const SubIterType &begin, const SubIterType &end
            ) {
                _iterators.emplace_back( begin, end );
            }

            /** The copy constructor. */
            ChainedIteratorsVector( const ChainedIteratorsVector &other ) {
                _iterators = other._iterators;
            }

            /** The move constructor. */
            ChainedIteratorsVector( ChainedIteratorsVector &&other ) {
                _iterators = std::move( other._iterators );
            }

            /** The copy assignment operator. */
            ChainedIteratorsVector & operator=( const ChainedIteratorsVector &other ) {
                _iterators = other._iterators;
                return *this;
            }

            /** The move assignment operator. */
            ChainedIteratorsVector & operator=( ChainedIteratorsVector &&other ) {
                _iterators = std::move( other._iterators );
                return *this;
            }

            /** The destructor. */
            ~ChainedIteratorsVector() {}

            /** The push_back method */
            void push_back( const SubIterType &begin, const SubIterType &end ) {
                _iterators.emplace_back( begin, end );
            }

            template< typename SubIterContainer >
            void push_back( const SubIterContainer &container ) {
                push_back( container.cbegin(), container.cend() );
            }

            /** The emplace_back method */
            template< typename... Args >
            void emplace_back( Args&&... args ) {
                _iterators.emplace_back( std::forward< Args >( args )... );
            }

            /** The clear method */
            void clear() {
                _iterators.clear();
            }

            /** The begin method. */
            iterator begin() const {
                return iterators::ChainedIterators< SubIterType >(
                    _iterators
                );
            }

            iterator cbegin() const {
                return begin();
            }

            /** The end method. */
            iterator end() const {
                return iterators::ChainedIterators< SubIterType >(
                    _iterators,
                    _iterators.size(),
                    0UL
                );
            }

            iterator cend() const {
                return end();
            }

            /** The size method. */
            size_t size() const {
                size_t dist = 0;
                for( size_t i = 0; i < _iterators.size(); ++i ) {
                    dist += size( i );
                }
                return dist;
            }

                /** The size method for a sub-iterator. */
            size_t size( size_t i ) const {
                assert( i < _iterators.size() );
                return std::distance( _iterators[i].first, _iterators[i].second );
            }

    };

} // namespace grb::utils::containers

#endif // end ``_H_CHAINEDITERATORS''

