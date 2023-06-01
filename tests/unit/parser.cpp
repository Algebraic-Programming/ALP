
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

#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

#include "graphblas/SynchronizedNonzeroIterator.hpp"


/**
 * Read a list of edges from a graph dataset.
 * Assuming they are comments, this parser skips lines that start with # or %.
 * If the first non-comment line consists out of three integers, the first two
 * of those three integers shall be taken as the dimensions of the graph, while
 * the third integer shall be taken as the number of edges in the graph.
 *
 * @param[in]  filename     The name of the file to read from.
 * @param[in]  use_indirect If true, the nodes use an indirect labelling scheme,
 *                          otherwise the nodes are used directly and 1-based
 *                          indicing is assumed (MatrixMarket).
 * @param[in,out]  n        If not equal to SIZE_MAX on input, value reflects
 *                          the maximum number of distinct nodes (i.e the size
 *                          of the matrix).
 *                          If equal to SIZE_MAX on input, the parser will
 *                          attempt to derice the maximum number of distinct
 *                          nodes from the input file.
 * @param[out] nz           A pointer to the number of non-zero matrix elements
 *                          (i.e. the number of edges).
 * @param[out] I            The source nodes that make up each edge.
 * @param[out] J            The destination nodes that make up each edge.
 * @param[out] weights      An optional weight applied to each edge - allocated
 *                          but not initialised. Can be NULL in which case no
 *                          allocation shall take place either.
 */
bool readEdges(
	const std::string filename,
	const bool use_indirect,
	size_t * const n, size_t * const nz,
	size_t ** const I, size_t ** const J,
	double ** const weights
) {
	// find the number of edges in the input file i.e. the non-zeros in the weight
	// matrix
	*nz = 0;
	std::ifstream myfile;
	myfile.open( filename );
	bool leave_open = false;
	if( myfile.is_open() ) {
		std::string line;
		while( std::getline( myfile, line ) ) {
			// ignore comments
			if( line[ 0 ] == '#' ) {
				continue;
			}
			if( line[ 0 ] == '%' ) {
				continue;
			}
			// count integers on this line
			size_t num = 0, elems[ 3 ];
			std::istringstream iss( line );
			while( iss >> elems[ num++ ] )
				;
			num--;
			// 2 integers represents an edge
			if( num == 2 ) {
				( *nz )++;
				// 3 integers represents n*m matrix and number of edges
			} else {
				if( *n == SIZE_MAX ) {
					*n = elems[ 1 ];
					assert( *n == elems[ 0 ] );
				}
				*nz = elems[ 2 ];
				leave_open = true;
				break;
			}
		}
		if( ! leave_open ) {
			myfile.close();
		}
	}

	// allocate space for weight matrix and matrix iterators
	if( weights != NULL ) {
		*weights = new double[ *nz ];
	}
	*I = new size_t[ *nz ];
	*J = new size_t[ *nz ];

	// take each edge and derive the associated nodes - either a direct or indirect mapping
	bool success = true;
	if( !leave_open ) {
		myfile.open( filename );
	}
	if( myfile.is_open() ) {
		std::map< size_t, size_t > indirect;
		size_t edge = 0;
		size_t next_node = 0;
		std::string line;
		// iterate over each line
		while( std::getline( myfile, line ) && success ) {
			// ignore comments
			if( line[ 0 ] == '#' ) {
				continue;
			}
			if( line[ 0 ] == '%' ) {
				continue;
			}
			// count integers on this line
			size_t num = 0, elems[ 3 ];
			std::istringstream iss( line );
			while( iss >> elems[ num++ ] )
				;
			num--;
			// 3 integers represents n*m matrix and number of edges
			if( num == 3 ) {
				continue;
			}
			// 2 integers represents an edge
			if( edge >= *nz ) {
				success = false;
				continue;
			}
			size_t n1 = elems[ 0 ], n2 = elems[ 1 ];
			// if an indirect mapping then find the map to the existing nodes or generate new ones
			if( use_indirect ) {
				size_t indirect_n1, indirect_n2;
				if( indirect.find( n1 ) == indirect.end() ) {
					indirect[ n1 ] = next_node++;
				}
				indirect_n1 = indirect[ n1 ];
				if( indirect.find( n2 ) == indirect.end() ) {
					indirect[ n2 ] = next_node++;
				}
				indirect_n2 = indirect[ n2 ];
				if( indirect_n1 < *n && indirect_n2 < *n ) {
					( *I )[ edge ] = indirect_n1;
					( *J )[ edge ] = indirect_n2;
				} else {
					success = false;
				}
			} else {
				// 1-base correction
				(void)--n1;
				(void)--n2;
				// otherwise assume a direct node mapping
				if( n1 < *n && n2 < *n ) {
					( *I )[ edge ] = n1;
					( *J )[ edge ] = n2;
				} else {
					std::cerr << "Edge with coordinates " << n1 << ", " << n2 << " is out of range (" << *n << ")!" << std::endl;
					success = false;
				}
			}
			edge++;
		}
		myfile.close();
		// deallocate memory if there was an issue
		if( !success ) {
			std::cerr << "Error during file I/O!" << std::endl;
			if( weights != NULL ) {
				delete[] * weights;
			}
			delete[] * I;
			delete[] * J;
		}
	} else {
		std::cerr << "Unable to open file" << std::endl;
		return false;
	}

	return success;
}

#ifdef COMPARE
 #include <iostream>
 #include <map>
 #include <set>

 #include "graphblas/utils/parser.hpp"

int main( int argc, char ** argv ) {
	std::cout << "Functional test executable: " << argv[ 0 ] << "\n";

	if( argc != 2 ) {
		std::cerr << "please, give path to cit-HepTh.txt" << std::endl;
		std::cout << "Test FAILED" << std::endl;
		return 255;
	}

	int ret = 0;

	// a naive storage of the input matrix
	std::map< size_t, std::set< size_t > > A;

	// use utils parser
	try {
		const char * const dataset_file = argv[ 1 ];
		grb::utils::MatrixFileReader< void > citHepTh( dataset_file , false, true );

		// fill A
		for( const auto & nz : citHepTh ) {
			// try to find row in A
			auto row = A.find( nz.first );
			if( row == A.end() ) {
				// if not found, add new row with this nonzero
				// as sole content
				std::set< size_t > initial;
				initial.insert( nz.second );
				A[ nz.first ] = initial;
			} else {
				// add this nonzero to the row found
				row->second.insert( nz.second );
			}
		}

		// use direct parser
		size_t nz, *I, *J, n;
		n = 27770;
		const bool rc = readEdges( dataset_file, true, &n, &nz, &I, &J, NULL );
		if( ! rc ) {
			std::cerr << "Error in use of direct parser.\n";
			ret = 1;
		}
		// check nonzero count
		if( nz != citHepTh.nz() ) {
			std::cerr << "Direct parser nz count does not match util parser.\n";
			ret = 2;
		}

		/* The below tests for automatic derivation of number of vertices, but this is
		 * not supported by the SNAP data files
		n = SIZE_MAX;
		const bool rc2 = readEdges(
			dataset_file, true, &n,
			&nz, &I, &J, NULL
		);
		if( !rc2 ) {
			std::cerr << "Error in use of direct parser.\n";
			ret = 3;
		}
		//check vertex count
		if( n != 27770 ) {
			std::cerr << "Direct parser could not derive correctly the number of "
				<< "vertices: returned " << n << " instead of 27770.\n";
			ret = 4;
		}
		//check nonzero count
		if( nz != citHepTh.nz() ) {
			std::cerr << "Direct parser nz count does not match util parser.\n";
			ret = 5;
		}
		*/

		// check synchronised iterator
		auto synced_it = grb::internal::makeSynchronized( I, J, I + nz, J + nz );
		for( size_t k = 0; ret == 0 && k < nz; ++k, ++synced_it ) {
			if( I[ k ] != synced_it.i() ) {
				std::cerr << "Synchronised file iterator has mismatching row indices at "
					<< "position " << k << ": read " << synced_it.i() << " instead of "
					<< I[ k ] << "\n";
				ret = 10;
			}
			if( J[ k ] != synced_it.j() ) {
				std::cerr << "Synchronised file iterator has mismatching column indices at "
					<< "position " << k << ": read " << synced_it.j() << " instead of "
					<< J[ k ] << "\n";
				ret = 11;
			}
		}
		// another nonzero count test
		nz = 0;
		for( const auto &row : A ) {
			nz += row.second.size();
		}
		if( nz != citHepTh.nz() ) {
			std::cerr << "Util parsers imported into std::map< size_t, std::set< size_t > > "
				<< "changes nonzero count ( " << nz << " versus " << citHepTh.nz()
				<< " ).\n";
			ret = 20;
		}

		// use non-maximal util parser
		grb::utils::MatrixFileReader< void > citHepTh2( dataset_file, false, true );

		if(
			citHepTh.filename() != citHepTh2.filename() ||
			citHepTh.m() != citHepTh2.m() ||
			citHepTh.n() != citHepTh2.n() || citHepTh.nz() != citHepTh2.nz() ||
			citHepTh.isPattern() != citHepTh2.isPattern() ||
			citHepTh.isSymmetric() != citHepTh2.isSymmetric() ||
			citHepTh.usesDirectAddressing() != citHepTh2.usesDirectAddressing()
		) {
			std::cerr << "Inferred matrix properties do not match explicitly given "
				<< "matrix properties.\n";
			ret = 30;
		}

		// check contents
		if( citHepTh.rowMap().size() != citHepTh2.rowMap().size() ) {
			std::cerr << "Inferred matrix and explicit matrix row maps are not of equal "
				<< "size (" << citHepTh.rowMap().size() << " vs. "
				<< citHepTh2.rowMap().size() << ").\n";
			ret = 33;
		};
		if( citHepTh.colMap().size() != citHepTh2.colMap().size() ) {
			std::cerr << "Inferred matrix and explicit matrix col maps are not of equal "
				<< "size (" << citHepTh.colMap().size() << " vs. "
				<< citHepTh2.colMap().size() << ").\n";
			ret = 36;
		};

		nz = 0;
		for( const auto &nonzero : citHepTh2 ) {
			(void) nonzero;
			(void) ++nz;
		}
		if( nz != citHepTh.nz() ) {
			std::cerr << "Inferred matrix does not contain all nonzeroes "
				<< "found in the explicit matrix.\n";
			ret = 40;
		}

 // the below test is incorrect since reordering of input changes indirect
 // mapping
 #if 0
		//check whether all nonzeroes are here
		for( size_t i = 0; i < nz; ++i ) {
			const auto row = A.find( I[ i ] );
			if( row == A.end() ) {
				std::cerr << "Row " << I[i] << " not found in util-parsed matrix.\n";
				ret = 3;
				break;
			}
			const auto col = row->second.find( J[ i ] );
			if( col == row->second.end() ) {
				std::cerr << "Nonzero at (" << I[i] << ", " << J[i] << ") not found in "
					<< "util-parsed matrix.\n";
				ret = 4;
				break;
			}
		}
 #endif

	} catch( std::runtime_error &e ) {
		std::cerr << "Caught exception: " << e.what() << std::endl;
		ret = 50;
	}

	// done
	std::cerr << std::flush;
	if( ret == 0 ) {
		std::cout << "Test OK\n" << std::endl;
	} else {
		std::cout << "Test FAILED\n" << std::endl;
	}
	return ret;
}
#endif

