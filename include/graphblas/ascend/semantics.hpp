#ifndef _H_ALP_ASCEND_SEMANTICS
#define _H_ALP_ASCEND_SEMANTICS

namespace alp {

	namespace internal {

		bool invalidForEachAxes( const std::vector< int > &axes );
		bool invalidAxes( const std::vector< int > &axes );

	} // namespace internal

} // namespace alp

#endif // _H_ALP_ASCEND_SEMANTICS
