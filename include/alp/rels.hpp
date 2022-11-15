
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
 * @author D. G. Spampinato
 * @date 3rd of November, 2022
 */

#ifndef _H_ALP_RELATIONS
#define _H_ALP_RELATIONS

#include <type_traits>

#include "type_traits.hpp"
#include "internalrels.hpp"

namespace alp {

	/**
	 * This namespace holds various standard operators such as #alp::relations::lt.
	 */
	namespace relations {

		/**
		 * This class implements the less-than relation.
		 * It exposes the complete interface detailed in 
		 * \a alp::relations::internal::HomogeneousRelation.
		 *
		 * \note A proper GraphBLAS program never uses the interface exposed by this
		 *       relation directly, and instead simply passes the relation on to
		 *       GraphBLAS functions.
		 *
		 * @tparam SET The domain and codomain of the relation.
		 *
		 * \warning This operator expects a numerical type for \a SET or types 
		 *          that have the appropriate operator<-functions available.
		 */
		// [Relation Wrapping]
		template< typename SET, enum Backend implementation = config::default_backend >
		class lt : public internal::HomogeneousRelation< internal::lt< SET, implementation > > {

			public:

				lt() {}
		};
		// [Relation Wrapping]

		/**
		 * This class implements the greater-than relation.
		 * It exposes the complete interface detailed in 
		 * \a alp::relations::internal::HomogeneousRelation.
		 *
		 * \note A proper GraphBLAS program never uses the interface exposed by this
		 *       relation directly, and instead simply passes the relation on to
		 *       GraphBLAS functions.
		 *
		 * @tparam SET The domain and codomain of the relation.
		 *
		 * \warning This operator expects a numerical type for \a SET or types 
		 *          that have the appropriate operator>-functions available.
		 */
		template< typename SET, enum Backend implementation = config::default_backend >
		class gt : public internal::HomogeneousRelation< internal::gt< SET, implementation > > {

			public:

				gt() {}
		};

		/**
		 * This class implements the equality relation.
		 * It exposes the complete interface detailed in 
		 * \a alp::relations::internal::HomogeneousRelation.
		 *
		 * \note A proper GraphBLAS program never uses the interface exposed by this
		 *       operator directly, and instead simply passes the operator on to
		 *       GraphBLAS functions.
		 *
		 * @tparam SET The domain and codomain of the relation.
		 *
		 * \warning This operator expects a numerical type for \a SET or types 
		 *          that have the appropriate operator==-functions available.
		 */
		template< typename SET, enum Backend implementation = config::default_backend >
		class eq : public internal::HomogeneousRelation< internal::eq< SET, implementation > > {

			public:

				eq() {}
		};

		/**
		 * This class implements the not-equal relation.
		 * It exposes the complete interface detailed in 
		 * \a alp::relations::internal::HomogeneousRelation.
		 *
		 * \note A proper GraphBLAS program never uses the interface exposed by this
		 *       operator directly, and instead simply passes the operator on to
		 *       GraphBLAS functions.
		 *
		 * @tparam SET The domain and codomain of the relation.
		 *
		 * \warning This operator expects a numerical type for \a SET or types 
		 *          that have the appropriate operator==-functions available.
		 */
		template< typename SET, enum Backend implementation = config::default_backend >
		class neq : public internal::HomogeneousRelation< internal::neq< SET, implementation > > {

			public:

				neq() {}
		};

		/**
		 * This class implements the less-than-or-equal relation.
		 * It exposes the complete interface detailed in 
		 * \a alp::relations::internal::HomogeneousRelation.
		 *
		 * \note A proper GraphBLAS program never uses the interface exposed by this
		 *       operator directly, and instead simply passes the operator on to
		 *       GraphBLAS functions.
		 *
		 * @tparam SET The domain and codomain of the relation.
		 *
		 * \warning This operator expects a numerical type for \a SET or types 
		 *          that have the appropriate operator<=-functions available.
		 */
		template< typename SET, enum Backend implementation = config::default_backend >
		class le : public internal::HomogeneousRelation< internal::le< SET, implementation > > {

			public:

				le() {}
		};

		/**
		 * This class implements the greater-than-or-equal relation.
		 * It exposes the complete interface detailed in 
		 * \a alp::relations::internal::HomogeneousRelation.
		 *
		 * \note A proper GraphBLAS program never uses the interface exposed by this
		 *       operator directly, and instead simply passes the operator on to
		 *       GraphBLAS functions.
		 *
		 * @tparam SET The domain and codomain of the relation.
		 *
		 * \warning This operator expects a numerical type for \a SET or types 
		 *          that have the appropriate operator<=-functions available.
		 */
		template< typename SET, enum Backend implementation = config::default_backend >
		class ge : public internal::HomogeneousRelation< internal::ge< SET, implementation > > {

			public:

				ge() {}
		};

	} // namespace relations

	// [Relation Type Traits]
	template<
		typename IntRel,
		enum Backend implementation
	>
	struct is_relation< relations::lt< IntRel, implementation > > {
		static const constexpr bool value = true;
	};
	// [Relation Type Traits]

	template<
		typename IntRel,
		enum Backend implementation
	>
	struct is_relation< relations::gt< IntRel, implementation > > {
		static const constexpr bool value = true;
	};

	template<
		typename IntRel,
		enum Backend implementation
	>
	struct is_relation< relations::eq< IntRel, implementation > > {
		static const constexpr bool value = true;
	};

	template<
		typename IntRel,
		enum Backend implementation
	>
	struct is_relation< relations::neq< IntRel, implementation > > {
		static const constexpr bool value = true;
	};

	template<
		typename IntRel,
		enum Backend implementation
	>
	struct is_relation< relations::le< IntRel, implementation > > {
		static const constexpr bool value = true;
	};

	template<
		typename IntRel,
		enum Backend implementation
	>
	struct is_relation< relations::ge< IntRel, implementation > > {
		static const constexpr bool value = true;
	};

} // namespace alp

#endif // end ``_H_ALP_RELATIONS''

