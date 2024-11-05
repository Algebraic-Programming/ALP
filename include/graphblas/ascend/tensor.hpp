#ifndef _H_ALP_ASCEND_TENSOR
#define _H_ALP_ASCEND_TENSOR

#include <functional>
#include <limits>
#include <cstddef>
#include <set>

#include <graphblas.hpp>

#include <graphblas/ascend/utils.hpp>
#include <graphblas/ascend/operators.hpp>


namespace alp {

	/**
	 * A global ALP/Ascend vector that resides in Ascend memory.
	 */

	class Tensor {

		private:

			size_t id;
			std::string name;
			Datatype type;
			internal::Scope scope;
			std::vector< int > axes;

			Tensor& access( const std::vector<int> &axes ) noexcept {
				(void) axes;
				return *this;
			}


		public:

			/** Maintains a counter for unique tensor names. */
			static size_t tensor_id;

			bool operator==( const Tensor &t ) const;
			bool operator!=( const Tensor &t ) const { return not ( *this == t ); }

			/**
			 * @deprecated 
			 * 
			 * @brief Tensor[i] operator is deprecated. Use Tensor(i, ...) instead
			 */
			template< typename T, typename U >
			T operator[]( const U axis ) = delete;

			/**
			 * @brief Replacement for Tensor[i] operator, allows to specify multiple 
			 * axes in any order.
			 */
			template< typename AnyType >
			Tensor& operator()( const AnyType &axis ) {
				std::vector<int> computedAxes{ getAxisId( axis ) };
				return access( computedAxes );
			}

			/**
			 * @brief Replacement for Tensor[i] operator, allows to specify multiple 
			 * axes in any order.
			 */
			template< typename AnyType, typename... AnyPackType >
			Tensor& operator()( const AnyType &axis, AnyPackType const... args ) {
				std::vector<int> computedAxes{ getAxisId( axis ) };
				for( auto arg : { args... } ) {
					computedAxes.push_back( getAxisId( arg ) );
				}
				return access( computedAxes );
			}

			/**
			 * @brief Assignment operator of a Tensor (deleted)
			 */
			void operator=( const Tensor& ) = delete;

			/**
			 * @brief Assignment operator of ReductionOperation
			 */
			void operator=( const ReductionOperation &op );

			/**
			 * @brief Assignment operator of ApplyOperation
			 */
			void operator=( const ApplyOperation &op );

			Tensor() = default;
			Tensor( const Tensor &view_parent, const std::vector< int > &_axes ) noexcept;
			Tensor( const Tensor &t ) noexcept;
			Tensor( const std::vector< int > &_axes, const Datatype type ) noexcept;

			Tensor(
				const Datatype type,
				const std::vector< int > &axes_vector
			) noexcept;

			virtual ~Tensor() noexcept { }

			size_t getID() const;
			const std::string &getName() const;
			alp::Datatype getType() const;
			internal::Scope getScope() const;
			const std::vector< int > &getAxes() const;
			bool isGlobalDecl() const;
			bool isLocalDecl() const;
			bool isTempDecl() const;

			std::string getAccessedElement( size_t id ) const;
			std::string getAscendName( size_t id ) const;
			std::string getAscendGlobalName( size_t id ) const;
			std::string getTQueBufName( size_t id ) const;
	};
}


template<>
struct std::hash< alp::Tensor >
{
	std::size_t operator()( const alp::Tensor& tensor ) const noexcept
	{
	    return std::hash< std::string >{}( tensor.getName() );
		
	}
};

#endif
